import threading
import logging
from typing import List, Callable, Iterator, Tuple, Dict

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
        max_workers: int = 2,
        enable_checkpointing: bool = False,
        checkpoints_dir: str = './checkpoints'
    ):
        self.datastore = DataStore(trials)
        self.adaptive_scheduler = adaptive_scheduler
        self.trial_updated_callback = trial_updated_callback
        self.insights_callback = insights_callback
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

    def start(self):
        """Starts the main execution loop in a background thread."""
        if not self._is_running:
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

    def _execution_loop(self):
        """
        The main loop that drives the experiment. It gets work from the adaptive
        scheduler, sends it to the compute scheduler, processes results, and
        determines the next batch of work.
        """
        logger.info("Runtime engine execution loop started.")

        # Get the first batch of work
        work_queue = self.adaptive_scheduler.get_initial_work_units(self.datastore.get_all_trials())

        while self._is_running and work_queue:
            # Sort the work queue based on trial priority before submitting it.
            # Higher priority values are processed first.
            work_queue.sort(
                key=lambda wu: self.datastore.get_trial(wu.trial_id).priority,
                reverse=True
            )

            results_iterator = self.scheduler.run(work_queue)
            work_queue = [] # Reset for the next batch

            for work_unit, result in results_iterator:
                if not self._is_running:
                    break # Exit if stop() was called during iteration

                trial = self.datastore.get_trial(work_unit.trial_id)
                if not trial:
                    logger.warning(f"Could not find trial {work_unit.trial_id} for completed work unit.")
                    continue

                # --- Update Trial State ---
                if 'error' in result:
                    logger.error(f"Work unit {work_unit.type} for trial {trial.id} failed: {result['error']}")
                    # Optionally, mark trial as FAILED
                else:
                    if work_unit.type == WorkUnitType.TRAIN_EPOCH:
                        # The worker returns metrics and state updates
                        trial.current_epoch = result['state_updates']['current_epoch']
                        trial.checkpoint_path = result['state_updates']['checkpoint_path']
                        for metric_name, value in result['metrics'].items():
                            trial.results.setdefault(metric_name, []).append((trial.current_epoch, value))

                    elif work_unit.type == WorkUnitType.PROFILE_SPEED:
                        # This part of the schema might need revisiting, assuming a simple 'time' key for now
                        trial.est_time_per_epoch = result.get('time')

                # Notify orchestrator about the update
                if self.trial_updated_callback:
                    self.trial_updated_callback(trial.to_dict())

                # Analyze for insights
                new_insights = self.insight_engine.analyze(trial)
                if new_insights and self.insights_callback:
                    self.insights_callback([insight.__dict__ for insight in new_insights])

                # Get next work units from the adaptive scheduler based on this result
                next_work = self.adaptive_scheduler.get_next_work_units(trial, self.datastore.get_all_trials())
                work_queue.extend(next_work)

        self._is_running = False
        logger.info("Runtime engine execution loop finished.")

    def submit_work(self, work_units: List[WorkUnit]) -> Iterator[Tuple[WorkUnit, dict]]:
        """Submits a list of work units to the scheduler and yields results."""
        if not self._is_running:
            raise RuntimeError("Runtime Engine is not running.")
        return self.scheduler.run(work_units)
