import threading
import logging
import time
from typing import List, Callable, Iterator, Tuple, Dict, Optional

from sde.core.types import Trial, WorkUnit, WorkUnitType, TrialStatus
from sde.engine.datastore import DataStore
from sde.engine.insight import Insight
from sde.engine.scheduler import Scheduler
from sde.engine.insight import InsightEngine
from sde.exploration.schedulers import AdaptiveScheduler

logger = logging.getLogger(__name__)

class SdeRuntimeEngine:
    """
    Wraps the core computational components (Scheduler, DataStore, etc.)
    and exposes a simple API to the Orchestrator. This is the "Engine Room".
    It runs the main experiment loop in a separate thread.
    """

    def __init__(
        self,
        trials: List[Trial],
        dataset_name: str,
        adaptive_scheduler: AdaptiveScheduler,
        trial_updated_callback: Callable[[Dict], None],
        insights_callback: Callable[[List[Dict]], None],
        run_completed_callback: Callable[[], None],
        patience_budget: Optional[Dict[str, int]] = None,
        max_workers: int = 2,
        enable_checkpointing: bool = False,
        checkpoints_dir: str = './checkpoints'
    ):
        self.datastore = DataStore(trials)
        self.adaptive_scheduler = adaptive_scheduler
        self.trial_updated_callback = trial_updated_callback
        self.insights_callback = insights_callback
        self.run_completed_callback = run_completed_callback
        self.patience_budget = patience_budget or {}
        self.insight_engine = InsightEngine(
            self.datastore.get_all_trials(),
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing
        )
        self.scheduler = Scheduler(
            datastore=self.datastore,
            dataset_name=dataset_name,
            max_workers=max_workers,
            enable_checkpointing=enable_checkpointing,
            checkpoints_dir=checkpoints_dir
        )
        self._is_running = False
        self._thread = None
        self._is_resuming = False
        self.work_queue: List[WorkUnit] = []
        self.start_time: Optional[float] = None

    def start(self, is_resuming: bool = False):
        """Starts the main execution loop in a background thread."""
        if not self._is_running:
            if not self.start_time:
                self.start_time = time.time()
            self._is_resuming = is_resuming
            self._is_running = True
            self.scheduler.start()
            self._thread = threading.Thread(target=self._execution_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stops the execution loop and the underlying scheduler."""
        if self._is_running:
            self._is_running = False
            self.scheduler.stop()
            if self._thread and self._thread.is_alive():
                self._thread.join()
            self._thread = None

    def cancel_work_for_trial(self, trial_id: str):
        """Passes a cancellation request down to the scheduler."""
        logger.info(f"Runtime engine received request to cancel work for trial {trial_id}.")
        self.scheduler.cancel_work_for_trial(trial_id)

    def add_trial_live(self, trial: Trial):
        """Injects a new trial into the live datastore."""
        # This assumes the datastore has a thread-safe method to add a trial.
        self.datastore.add_trial(trial)
        logger.info(f"Added new trial {trial.id} to the live datastore.")

    def inject_trial(self, trial: Trial):
        """
        Injects a new trial into a live run.
        Adds the trial to the datastore and queues its first work unit.
        """
        logger.info(f"Injecting new trial {trial.id} into live run.")
        # Set status to ACTIVE since we are creating work for it immediately
        trial.status = TrialStatus.ACTIVE
        self.datastore.add_trial(trial)

        # Create the first work unit for the new trial.
        initial_work_unit = WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)

        # Add the work unit to the live work queue. This is thread-safe enough
        # as list.append is atomic and the consumer loop iterates on a copy.
        self.work_queue.append(initial_work_unit)

    def _execution_loop(self):
        """
        The main loop that drives the experiment. It populates the work queue,
        processes work, and checks against the budget until the run is stopped,
        the queue is empty, or the budget is exhausted.
        """
        logger.info("Runtime engine execution loop started.")
        run_finished_naturally = False

        try:
            if not self.work_queue:
                if self._is_resuming:
                    logger.info("Resuming run. Generating work for active trials.")
                    active_trials = [t for t in self.datastore.get_all_trials() if t.status == TrialStatus.ACTIVE]
                    self.work_queue = [WorkUnit(trial_id=t.id, type=WorkUnitType.TRAIN_EPOCH) for t in active_trials]
                else:
                    logger.info("Fresh run. Getting initial work units.")
                    self.work_queue = self.adaptive_scheduler.get_initial_work_units(self.datastore.get_all_trials())

            while self._is_running:
                # --- Budget and Termination Checks ---
                if not self.work_queue:
                    logger.info("Work queue is empty. Execution loop is finishing.")
                    run_finished_naturally = True
                    break

                max_trials = self.patience_budget.get('max_trials')
                if max_trials and len(self.datastore.get_all_trials()) >= max_trials:
                    logger.info(f"Budget exhausted: reached max_trials ({max_trials}).")
                    run_finished_naturally = True
                    break

                max_time_mins = self.patience_budget.get('max_time_mins')
                if max_time_mins and (time.time() - self.start_time) >= max_time_mins * 60:
                    logger.info(f"Budget exhausted: reached max_time_mins ({max_time_mins}).")
                    run_finished_naturally = True
                    break

                # --- Work Processing ---
                self.work_queue.sort(key=lambda wu: self.datastore.get_trial(wu.trial_id).priority, reverse=True)
                current_batch = self.work_queue
                self.work_queue = []
                results_iterator = self.scheduler.run(current_batch)

                for work_unit, result in results_iterator:
                    if not self._is_running:
                        break

                    trial = self.datastore.get_trial(work_unit.trial_id)
                    if not trial:
                        logger.warning(f"Could not find trial {work_unit.trial_id} for completed work unit.")
                        continue

                    if 'error' in result:
                        logger.error(f"Work unit {work_unit.type} for trial {trial.id} failed: {result['error']}")
                    else:
                        if work_unit.type == WorkUnitType.TRAIN_EPOCH:
                            trial.current_epoch = result['state_updates']['current_epoch']
                            trial.checkpoint_path = result['state_updates']['checkpoint_path']
                            for metric, value in result['metrics'].items():
                                trial.results.setdefault(metric, []).append((trial.current_epoch, value))
                        elif work_unit.type == WorkUnitType.PROFILE_SPEED:
                            trial.est_time_per_epoch = result.get('time')

                    if self.trial_updated_callback:
                        self.trial_updated_callback(trial.to_dict())

                    new_insights = self.insight_engine.analyze(trial)
                    if new_insights and self.insights_callback:
                        self.insights_callback([insight.__dict__ for insight in new_insights])

                    next_work = self.adaptive_scheduler.get_next_work_units(trial, self.datastore.get_all_trials())
                    self.work_queue.extend(next_work)

                if not self._is_running: # Check if stop was called during result iteration
                    break

        finally:
            self._is_running = False
            logger.info("Runtime engine execution loop finished.")
            if run_finished_naturally and self.run_completed_callback:
                self.run_completed_callback()

    def submit_work(self, work_units: List[WorkUnit]) -> Iterator[Tuple[WorkUnit, dict]]:
        """Submits a list of work units to the scheduler and yields results."""
        if not self._is_running:
            raise RuntimeError("Runtime Engine is not running.")
        return self.scheduler.run(work_units)
