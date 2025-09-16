"""Module for the SDE runtime engine, the computational core."""
import logging
import queue
import threading
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from sde.core.comms import EngineCommand
from sde.core.comms import EngineEvent
from sde.core.domain import Challenge
from sde.core.domain import ExecutionSettings
from sde.core.domain import PatienceBudget
from sde.core.domain import Trial
from sde.core.domain import TrialStatus
from sde.core.domain import WorkUnit
from sde.exploration.schedulers import AdaptiveScheduler

from .compute_scheduler import ComputeScheduler
from .datastore import DataStore
from .factory import SchedulerFactory
from .insight import InsightEngine

logger = logging.getLogger(__name__)


class RuntimeStatus(Enum):
    """Define the lifecycle status of the runtime engine."""

    IDLE = "IDLE"  # Not running, waiting for a START command.
    RUNNING = "RUNNING"  # Actively processing work.
    PAUSED = "PAUSED"  # The main loop is paused, no new work is dispatched.
    STOPPING = "STOPPING"  # A stop has been requested, finishing in-flight work.
    STOPPED = "STOPPED"  # The engine thread has been shut down.


class SdeRuntimeEngine:
    """Run SDE experiments in a self-contained, command-driven engine.

    It owns all the core computational components and runs the main experiment
    loop in a separate thread. It receives commands from an external proxy
    and emits events via a callback.
    """

    def __init__(self, event_callback: Callable[[EngineEvent, Dict[str, Any]], None]):
        """Initialize the SdeRuntimeEngine."""
        # Core components are initialized on START_RUN
        self.datastore: Optional[DataStore] = None
        self.adaptive_scheduler: Optional[AdaptiveScheduler] = None
        self.insight_engine: Optional[InsightEngine] = None
        self.compute_scheduler: Optional[ComputeScheduler] = None

        # --- Threading and State Control ---
        self.command_queue: queue.Queue[Tuple[EngineCommand, Dict[str, Any]]] = (
            queue.Queue()
        )
        self.work_queue: queue.PriorityQueue[Tuple[int, int, WorkUnit]] = (
            queue.PriorityQueue()
        )
        self._work_counter = 0  # Tie-breaker for priority queue
        self._pause_event = threading.Event()
        self._cancelled_trials: set[str] = set()
        self._thread: Optional[threading.Thread] = None
        self._status = RuntimeStatus.IDLE
        self.event_callback = event_callback

    def _emit_event(self, event_type: EngineEvent, payload: Dict[str, Any]):
        """Emit an event to the orchestrator layer."""
        self.event_callback(event_type, payload)

    def post_command(self, command: EngineCommand, payload: Dict[str, Any]) -> None:
        """Add a command to the command queue for the engine to process."""
        self.command_queue.put((command, payload))

    def _initialize_runtime(
        self,
        trials: List[Trial],
        challenge: Challenge,
        policy_name: str,
        patience_budget: PatienceBudget,
        execution_settings: ExecutionSettings,
    ):
        """Set up all the core components based on configuration data."""
        self.datastore = DataStore(trials)
        self.adaptive_scheduler = SchedulerFactory.create_scheduler(
            policy_name=policy_name,
            challenge_name=challenge.name,
            patience_budget=patience_budget.to_dict(),
        )
        self.insight_engine = InsightEngine(
            self.datastore.get_all_trials(),
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing,
        )
        self.compute_scheduler = ComputeScheduler(
            datastore=self.datastore,
            dataset_name=challenge.name,
            max_workers=execution_settings.num_workers,
            enable_checkpointing=execution_settings.enable_checkpointing,
            work_unit_timeout=execution_settings.work_unit_timeout_seconds,
        )
        logger.info(f"Runtime engine initialized with {policy_name} scheduler.")

    def _put_work_in_queue(self, work_unit: WorkUnit):
        """Add a work unit to the priority queue with a tie-breaker."""
        assert self.datastore is not None
        trial = self.datastore.get_trial(work_unit.trial_id)
        if trial:
            priority = -trial.priority  # Negated for min-heap
            self.work_queue.put((priority, self._work_counter, work_unit))
            self._work_counter += 1

    def _initialize_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the START_RUN payload and initialize all core components."""
        exp_def = payload.get("experiment_definition", {})
        exec_settings_dict = payload.get("execution_settings")

        trials = [Trial.from_dict(t) for t in exp_def.get("trials", {}).values()]
        execution_settings = (
            ExecutionSettings.from_dict(exec_settings_dict)
            if exec_settings_dict
            else ExecutionSettings()
        )
        challenge = Challenge.from_dict(exp_def.get("challenge"))
        patience_budget_dict = exp_def.get("patience_budget")
        patience_budget = (
            PatienceBudget.from_dict(patience_budget_dict)
            if patience_budget_dict is not None
            else PatienceBudget()
        )

        self._initialize_runtime(
            trials=trials,
            challenge=challenge,
            policy_name=exp_def.get("adaptive_policy"),
            patience_budget=patience_budget,
            execution_settings=execution_settings,
        )
        return exp_def

    def _handle_start_run(self, payload: Dict[str, Any]):
        """Handle the START_RUN command."""
        if self._status != RuntimeStatus.IDLE:
            msg = f"START_RUN received, but engine is in state {self._status.name}."
            logger.warning(msg)
            return

        # 1. Initialize all core components from the payload
        exp_def = self._initialize_from_payload(payload)
        start_paused = payload.get("start_paused", False)

        assert self.datastore is not None
        assert self.adaptive_scheduler is not None
        assert self.compute_scheduler is not None

        # 2. Get initial work units using the new stateless scheduler interface
        all_trials = self.datastore.get_all_trials()
        is_resumed_run = any(
            t.status != TrialStatus.PENDING for t in all_trials.values()
        )

        if is_resumed_run:
            logger.info("Resuming experiment. Rehydrating work queue...")
            work_units = self.adaptive_scheduler.rehydrate_work_units(all_trials)
        else:
            logger.info("Starting fresh experiment. Generating initial work units...")
            # The scheduler may now mutate the trial objects (e.g., add tags)
            pending_trials = [
                t for t in all_trials.values() if t.status == TrialStatus.PENDING
            ]
            work_units, scheduler_state = (
                self.adaptive_scheduler.get_initial_work_units(
                    pending_trials, exp_def.get("scheduler_state", {})
                )
            )
            exp_def["scheduler_state"] = scheduler_state

        for work_unit in work_units:
            self._put_work_in_queue(work_unit)

        # 3. Start the compute scheduler and the main execution loop
        self.compute_scheduler.start()
        self._status = RuntimeStatus.PAUSED if start_paused else RuntimeStatus.RUNNING
        if not start_paused:
            self._pause_event.set()

        self._thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._thread.start()
        logger.info("SdeRuntimeEngine started successfully.")
        # The payload now only contains the data the orchestrator doesn't know:
        # the initial state of the trials created by the adaptive scheduler.
        initial_trials_dict = [
            t.to_dict() for t in self.datastore.get_all_trials().values()
        ]
        self._emit_event(EngineEvent.RUN_STARTED, {"trials": initial_trials_dict})

    def _handle_stop_run(self, payload: Dict[str, Any]):
        """Handle the STOP_RUN command."""
        if self._status in [RuntimeStatus.STOPPING, RuntimeStatus.STOPPED]:
            return

        self._status = RuntimeStatus.STOPPING
        self._pause_event.set()  # Ensure loop isn't blocked on pause
        if self.compute_scheduler:
            self.compute_scheduler.stop()

        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None
        self._status = RuntimeStatus.STOPPED
        logger.info("SdeRuntimeEngine has been cleanly shut down.")
        self._emit_event(EngineEvent.RUN_STOPPED, {})

    def _handle_pause_run(self, payload: Dict[str, Any]):
        """Handle the PAUSE_RUN command."""
        if self._status == RuntimeStatus.RUNNING:
            self._status = RuntimeStatus.PAUSED
            self._pause_event.clear()
            logger.info("Runtime engine execution paused.")
            self._emit_event(EngineEvent.RUN_PAUSED, {})

    def _handle_resume_run(self, payload: Dict[str, Any]):
        """Handle the RESUME_RUN command."""
        if self._status == RuntimeStatus.PAUSED:
            self._status = RuntimeStatus.RUNNING
            self._pause_event.set()
            logger.info("Runtime engine execution resumed.")
            self._emit_event(EngineEvent.RUN_RESUMED, {})

    def cancel_work_for_trial(self, trial_id: str) -> None:
        """Mark a trial as cancelled and stop its work in the compute scheduler."""
        logger.info(f"Cancelling work for trial {trial_id}.")
        self._cancelled_trials.add(trial_id)
        if self.compute_scheduler:
            self.compute_scheduler.cancel_work_for_trial(trial_id)

    def _process_command_queue(self):
        """Process all pending commands in the queue."""
        try:
            while not self.command_queue.empty():
                command, payload = self.command_queue.get_nowait()
                handler_name = f"_handle_{command.value.lower()}"
                handler = getattr(self, handler_name, None)
                if handler:
                    handler(payload)
                else:
                    logger.warning(f"Unknown command received: {command.name}")
        except queue.Empty:
            return  # Should not happen, but for safety

    def _execution_loop(self):
        """Run the main loop that drives the experiment."""
        logger.info("Runtime engine execution loop started.")
        while self._status not in [RuntimeStatus.STOPPING, RuntimeStatus.STOPPED]:
            self._pause_event.wait()
            if self._status in [RuntimeStatus.STOPPING, RuntimeStatus.STOPPED]:
                break

            self._process_command_queue()

            current_batch = []
            while (
                not self.work_queue.empty()
                and len(current_batch) < self.compute_scheduler.max_workers
            ):
                try:
                    _, _, work_unit = self.work_queue.get_nowait()
                    if work_unit.trial_id in self._cancelled_trials:
                        logger.info(
                            "Discarding cancelled work for trial %s",
                            work_unit.trial_id,
                        )
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
                if self._status in [RuntimeStatus.STOPPING, RuntimeStatus.STOPPED]:
                    break
                self._process_completed_work_unit(work_unit, result)

        self._status = RuntimeStatus.STOPPED
        logger.info("Runtime engine execution loop finished.")

    def _process_completed_work_unit(self, work_unit: WorkUnit, result: dict):
        """Handle the result of a single completed work unit."""
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
                f"Trial {work_unit.trial_id} disappeared before result recorded."
            )
            return

        if updated_trial.id in self._cancelled_trials:
            logger.info(f"Ignoring result for cancelled trial {updated_trial.id}")
            return

        if "error" in result:
            msg = (
                f"Work unit {work_unit.type} for trial {updated_trial.id} "
                f"failed: {result['error']}"
            )
            logger.error(msg)
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
                    EngineEvent.INSIGHTS_GENERATED,
                    {"insights": [i.to_dict() for i in insights]},
                )

            if updated_trial.status == TrialStatus.ACTIVE:
                next_work_units = self.adaptive_scheduler.get_next_work_units(
                    updated_trial, self.datastore.get_all_trials()
                )
                for next_wu in next_work_units:
                    self._put_work_in_queue(next_wu)

        self._emit_event(EngineEvent.TRIAL_UPDATED, {"trial": updated_trial.to_dict()})
