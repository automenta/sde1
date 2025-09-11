import logging
import queue
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

from sde.core.domain import Experiment
from sde.core.domain import Trial
from sde.core.domain import TrialStatus
from sde.core.domain import WorkUnit
from ..exploration.schedulers import AdaptiveScheduler
from .compute_scheduler import ComputeScheduler
from .datastore import DataStore
from .factory import SchedulerFactory
from .insight import InsightEngine

logger = logging.getLogger(__name__)


class SdeRuntimeEngine:
    """The self-contained, command-driven engine for running SDE experiments.
    It owns all the core computational components and runs the main experiment
    loop in a separate thread. It receives commands from an external proxy
    and emits events via a callback.
    """

    def __init__(self, event_callback: Callable[[str, Dict[str, Any]], None]):
        """Initializes the SdeRuntimeEngine.

        Args:
            event_callback: A function to call to emit events to the outside world.
                The function should accept an event type (str) and a payload (dict).
        """
        # Core components are initialized on START_RUN
        self.experiment: Optional[Experiment] = None
        self.datastore: Optional[DataStore] = None
        self.adaptive_scheduler: Optional[AdaptiveScheduler] = None
        self.insight_engine: Optional[InsightEngine] = None
        self.compute_scheduler: Optional[ComputeScheduler] = None

        # --- Threading and State Control ---
        self.command_queue: queue.Queue[Tuple[str, Dict[str, Any]]] = queue.Queue()
        self.work_queue: queue.PriorityQueue[
            Tuple[int, int, WorkUnit]
        ] = queue.PriorityQueue()
        self._work_counter = 0  # Tie-breaker for priority queue
        self._pause_event = threading.Event()
        self._cancelled_trials: set[str] = set()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self.event_callback = event_callback

    def _emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Emits an event to the orchestrator layer."""
        self.event_callback(event_type, payload)

    def post_command(self, command: str, payload: Dict[str, Any]) -> None:
        """Adds a command to the command queue for the engine to process."""
        self.command_queue.put((command, payload))

    def _initialize_from_experiment(self, experiment: Experiment):
        """Sets up all the core components based on an experiment definition."""
        self.experiment = experiment
        self.datastore = DataStore(list(self.experiment.trials.values()))
        self.adaptive_scheduler = SchedulerFactory.create_scheduler(
            policy_name=self.experiment.adaptive_policy,
            challenge_name=self.experiment.challenge["name"],
            patience_budget=self.experiment.patience_budget,
        )
        self.insight_engine = InsightEngine(
            self.datastore.get_all_trials(),
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing,
        )
        self.compute_scheduler = ComputeScheduler(
            datastore=self.datastore,
            dataset_name=self.experiment.challenge["name"],
            max_workers=self.experiment.execution_settings.num_workers,
            enable_checkpointing=self.experiment.execution_settings.enable_checkpointing,
            work_unit_timeout=self.experiment.execution_settings.work_unit_timeout_seconds,
        )
        logger.info(f"Runtime engine initialized with {self.experiment.adaptive_policy} scheduler.")

    def _put_work_in_queue(self, work_unit: WorkUnit):
        """Adds a work unit to the priority queue with a tie-breaker."""
        assert self.datastore is not None
        trial = self.datastore.get_trial(work_unit.trial_id)
        if trial:
            priority = -trial.priority  # Negated for min-heap
            self.work_queue.put((priority, self._work_counter, work_unit))
            self._work_counter += 1

    def _handle_start_run(self, payload: Dict[str, Any]):
        """Handles the START_RUN command."""
        print("RUNTIME: _handle_start_run called")
        if self._thread is not None:
            logger.warning("START_RUN command received, but engine is already running.")
            return

        experiment_definition = payload.get("experiment_definition", {})
        start_paused = payload.get("start_paused", False)
        experiment = Experiment.from_dict(experiment_definition)

        # 1. Initialize all core components
        self._initialize_from_experiment(experiment)

        # Assertions to help mypy after initialization
        assert self.datastore is not None
        assert self.adaptive_scheduler is not None
        assert self.experiment is not None
        assert self.compute_scheduler is not None

        # 2. Get initial work units
        all_trials = self.datastore.get_all_trials()
        is_resumed_run = any(t.status != TrialStatus.PENDING for t in all_trials.values())

        if is_resumed_run:
            logger.info("Resuming experiment. Rehydrating work queue...")
            work_units = self.adaptive_scheduler.rehydrate_work_units(
                all_trials, self.experiment.scheduler_state
            )
        else:
            logger.info("Starting fresh experiment. Generating initial work units...")
            work_units, updated_scheduler_state = self.adaptive_scheduler.get_initial_work_units(
                all_trials, self.experiment.scheduler_state
            )
            if updated_scheduler_state:
                self.experiment.scheduler_state.update(updated_scheduler_state)

        for work_unit in work_units:
            self._put_work_in_queue(work_unit)

        # 3. Start the compute scheduler and the main execution loop
        self.compute_scheduler.start()
        self._is_running = True
        if not start_paused:
            self._pause_event.set()

        self._thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._thread.start()
        logger.info("SdeRuntimeEngine started successfully.")
        self._emit_event("RUN_STARTED", self.experiment.to_dict())

    def _handle_stop_run(self, payload: Dict[str, Any]):
        """Handles the STOP_RUN command."""
        if not self._is_running:
            return

        self._is_running = False
        self._pause_event.set()  # Ensure loop isn't blocked on pause
        if self.compute_scheduler:
            self.compute_scheduler.stop()

        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None
        logger.info("SdeRuntimeEngine has been cleanly shut down.")
        self._emit_event("RUN_STOPPED", {})

    def _handle_pause_run(self, payload: Dict[str, Any]):
        """Handles the PAUSE_RUN command."""
        if self._is_running:
            self._pause_event.clear()
            logger.info("Runtime engine execution paused.")
            self._emit_event("RUN_PAUSED", {})

    def _handle_resume_run(self, payload: Dict[str, Any]):
        """Handles the RESUME_RUN command."""
        if self._is_running:
            self._pause_event.set()
            logger.info("Runtime engine execution resumed.")
            self._emit_event("RUN_RESUMED", {})

    def cancel_work_for_trial(self, trial_id: str) -> None:
        """Marks a trial as cancelled and stops its work in the compute scheduler."""
        logger.info(f"Cancelling work for trial {trial_id}.")
        self._cancelled_trials.add(trial_id)
        if self.compute_scheduler:
            self.compute_scheduler.cancel_work_for_trial(trial_id)

    def _process_command_queue(self):
        """Processes all pending commands in the queue."""
        try:
            while not self.command_queue.empty():
                command, payload = self.command_queue.get_nowait()
                handler_name = f"_handle_{command.lower()}"
                handler = getattr(self, handler_name, None)
                if handler:
                    handler(payload)
                else:
                    logger.warning(f"Unknown command received: {command}")
        except queue.Empty:
            return  # Should not happen, but for safety

    def _execution_loop(self):
        """The main loop that drives the experiment."""
        logger.info("Runtime engine execution loop started.")
        while self._is_running:
            self._pause_event.wait()
            if not self._is_running:
                break

            self._process_command_queue()

            current_batch = []
            while not self.work_queue.empty() and len(current_batch) < self.compute_scheduler.max_workers:
                try:
                    _, _, work_unit = self.work_queue.get_nowait()
                    if work_unit.trial_id in self._cancelled_trials:
                        logger.info(f"Discarding cancelled work unit for trial {work_unit.trial_id}")
                        continue
                    current_batch.append(work_unit)
                except queue.Empty:
                    break

            if not current_batch:
                if all(
                    t.status in (TrialStatus.COMPLETED, TrialStatus.PRUNED)
                    for t in self.datastore.get_all_trials().values()
                ):
                    logger.info("All trials are completed or pruned. Shutting down.")
                    self._handle_stop_run({})
                    break
                threading.Event().wait(0.5)
                continue

            results_iterator = self.compute_scheduler.run(current_batch)
            for work_unit, result in results_iterator:
                self._pause_event.wait()
                if not self._is_running:
                    break
                self._process_completed_work_unit(work_unit, result)

        self._is_running = False
        logger.info("Runtime engine execution loop finished.")

    def _process_completed_work_unit(self, work_unit: WorkUnit, result: dict):
        """Handles the result of a single completed work unit."""
        assert self.datastore is not None
        assert self.insight_engine is not None
        assert self.adaptive_scheduler is not None

        trial = self.datastore.get_trial(work_unit.trial_id)
        if not trial:
            logger.warning(
                f"Could not find trial {work_unit.trial_id} to record result."
            )
            return
        original_status = trial.status

        updated_trial = self.datastore.record_work_unit_result(work_unit, result)
        if not updated_trial:
            # This can happen if the trial was cancelled and removed concurrently
            logger.warning(
                f"Trial {work_unit.trial_id} disappeared before result was recorded."
            )
            return

        if updated_trial.id in self._cancelled_trials:
            logger.info(f"Ignoring result for cancelled trial {updated_trial.id}")
            return

        if "error" in result:
            logger.error(
                f"Work unit {work_unit.type} for trial {updated_trial.id} failed: {result['error']}"
            )
            updated_trial.status = TrialStatus.FAILED
        else:
            is_newly_finished = updated_trial.status in (
                TrialStatus.COMPLETED,
                TrialStatus.PRUNED,
            ) and original_status not in (TrialStatus.COMPLETED, TrialStatus.PRUNED)
            insights = self.insight_engine.analyze_on_epoch(updated_trial)
            if is_newly_finished:
                logger.info(
                    f"Trial {updated_trial.id} has finished. Running final analysis."
                )
                insights.extend(self.insight_engine.analyze_on_finish(updated_trial))

            if insights:
                self._emit_event(
                    "INSIGHTS_GENERATED", {"insights": [i.__dict__ for i in insights]}
                )

            if updated_trial.status == TrialStatus.ACTIVE:
                next_work_units = self.adaptive_scheduler.get_next_work_units(
                    updated_trial, self.datastore.get_all_trials()
                )
                for next_wu in next_work_units:
                    self._put_work_in_queue(next_wu)

        self._emit_event("TRIAL_UPDATED", {"trial": updated_trial.to_dict()})

    def submit_work(
        self, work_units: List[WorkUnit]
    ) -> Iterator[Tuple[WorkUnit, Dict[str, Any]]]:
        """Submits a list of work units to the scheduler and yields results."""
        if not self._is_running or not self.compute_scheduler:
            raise RuntimeError("Runtime Engine is not running.")
        return self.compute_scheduler.run(work_units)  # type: ignore
