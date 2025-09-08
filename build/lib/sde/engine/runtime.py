import logging
import queue
import threading
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

from sde.core.types import ExecutionSettings
from sde.core.types import Experiment
from sde.core.types import Trial
from sde.core.types import TrialStatus
from sde.core.types import WorkUnit
from sde.engine.compute_scheduler import ComputeScheduler
from sde.engine.datastore import DataStore
from sde.engine.insight import InsightEngine
from sde.exploration.schedulers import AdaptiveScheduler

logger = logging.getLogger(__name__)


class SdeRuntimeEngine:
    """Wraps the core computational components (ComputeScheduler, DataStore, etc.)
    and exposes a simple API to the Orchestrator. This is the "Engine Room".
    It runs the main experiment loop in a separate thread.
    """

    def __init__(
        self,
        experiment: Experiment,
        adaptive_scheduler: AdaptiveScheduler,
        trial_updated_callback: Callable[[Dict], None],
        insights_callback: Callable[[List[Dict]], None],
        execution_settings: ExecutionSettings,
        checkpoints_dir: str = "./checkpoints",
    ):
        """Initializes the SdeRuntimeEngine.

        Args:
            experiment: The full Experiment object to run.
            adaptive_scheduler: The policy for scheduling work and pruning trials.
            trial_updated_callback: A function to call when a trial's state is updated.
            insights_callback: A function to call when new insights are generated.
            execution_settings: The execution settings for the run.
            checkpoints_dir: The directory to store model checkpoints.

        """
        self.experiment = experiment
        self.datastore = DataStore(list(experiment.trials.values()))
        self.adaptive_scheduler = adaptive_scheduler
        self.trial_updated_callback = trial_updated_callback
        self.insights_callback = insights_callback
        self.insight_engine = InsightEngine(
            self.datastore.get_all_trials(),
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing,
        )
        self.compute_scheduler = ComputeScheduler(
            datastore=self.datastore,
            dataset_name=experiment.challenge["name"],
            max_workers=execution_settings.num_workers,
            enable_checkpointing=execution_settings.enable_checkpointing,
            work_unit_timeout=execution_settings.work_unit_timeout_seconds,
            checkpoints_dir=checkpoints_dir,
        )
        # --- Threading and State Control ---
        self._is_running = False  # Flag to signal the main loop to terminate.
        self._thread: Optional[threading.Thread] = None  # The main execution thread.
        self.work_queue = queue.PriorityQueue()  # Thread-safe queue for pending work.
        self._work_counter = 0  # Tie-breaker for priority queue
        self._pause_event = threading.Event()  # Used to pause and resume the loop.
        self._cancelled_trials = set()  # A set of trial_ids to ignore.

    def _put_work_in_queue(self, work_unit: WorkUnit):
        """Adds a work unit to the priority queue with a tie-breaker."""
        trial = self.datastore.get_trial(work_unit.trial_id)
        priority = -trial.priority  # Negated for min-heap
        self.work_queue.put((priority, self._work_counter, work_unit))
        self._work_counter += 1

    def start(self, start_paused: bool = False) -> None:
        """Starts the main execution loop in a background thread."""
        if self._thread is None:
            self._is_running = True
            if not start_paused:
                self._pause_event.set()  # Start in a "not paused" state

            # --- Work Population ---
            # Check if this is a fresh run or a resumed run
            is_resumed_run = any(
                t.status != TrialStatus.PENDING for t in self.datastore.get_all_trials().values()
            )

            if is_resumed_run:
                logger.info("Resuming experiment. Rehydrating work queue...")
                work_units = self.adaptive_scheduler.rehydrate_work_units(
                    self.experiment
                )
            else:
                logger.info("Starting fresh experiment. Generating initial work units...")
                work_units = self.adaptive_scheduler.get_initial_work_units(
                    self.experiment
                )

            for work_unit in work_units:
                self._put_work_in_queue(work_unit)

            self.compute_scheduler.start()
            self._thread = threading.Thread(target=self._execution_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Signals the execution loop to stop and cleans up."""
        if self._is_running:
            self._is_running = False
            self._pause_event.set()  # Ensure loop isn't blocked on pause

            # Stop the compute scheduler first, which will shut down the process pool
            self.compute_scheduler.stop()

            # Now, wait for the main execution loop thread to finish
            if self._thread and self._thread.is_alive():
                self._thread.join()  # Wait indefinitely for a clean exit
            self._thread = None
            logger.info("SdeRuntimeEngine has been cleanly shut down.")

    def pause(self) -> None:
        """Pauses the execution loop."""
        if self._is_running:
            self._pause_event.clear()  # Clearing the event causes the loop to block
            logger.info("Runtime engine execution paused.")

    def resume(self) -> None:
        """Resumes the execution loop."""
        if self._is_running:
            self._pause_event.set()  # Setting the event unblocks the loop
            logger.info("Runtime engine execution resumed.")

    def cancel_work_for_trial(self, trial_id: str) -> None:
        """Passes a cancellation request down to the scheduler and marks the trial
        so any pending work units for it are ignored. This is thread-safe.
        """
        logger.info(
            f"Runtime engine received request to cancel work for trial {trial_id}."
        )
        self._cancelled_trials.add(trial_id)
        self.compute_scheduler.cancel_work_for_trial(trial_id)

    def add_trials_live(self, trials: List[Trial]) -> None:
        """Injects new trials into the live datastore and generates work units for them,
        adding them to the active work queue. This is thread-safe.
        """
        if not self._is_running:
            logger.warning("Cannot add trials live, engine is not running.")
            return

        for trial in trials:
            self.datastore.add_trial(trial)
            logger.info(f"Added new trial {trial.id} to the live datastore.")

        # Generate work units for the new trials.
        # The schedulers' get_initial_work_units methods find all PENDING trials
        # and create a TRAIN_EPOCH work unit. We can replicate that simple logic
        # here for just the new trials.
        new_work_units = []
        for trial in trials:
            if trial.status == TrialStatus.PENDING:
                trial.status = TrialStatus.ACTIVE
                new_work_units.append(
                    WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)
                )

        for work_unit in new_work_units:
            self._put_work_in_queue(work_unit)

        logger.info(
            f"Added {len(new_work_units)} new work units to the live queue."
        )

    def update_adaptive_policy(self, new_scheduler: AdaptiveScheduler) -> None:
        """Safely swaps the adaptive scheduler mid-run.
        This is a thread-safe operation.
        """
        if not self._is_running:
            logger.warning("Cannot update adaptive policy, engine is not running.")
            return

        # The key is that the adaptive_scheduler is only accessed inside the
        # _execution_loop after a batch of work is completed. We don't need
        # a lock here because the replacement is atomic. The loop will simply
        # pick up the new scheduler instance on its next iteration.
        self.adaptive_scheduler = new_scheduler
        logger.info(
            f"Adaptive policy has been hot-swapped to {new_scheduler.__class__.__name__}."
        )

    def _execution_loop(self):
        """The main loop that drives the experiment. It gets work from the adaptive
        scheduler, sends it to the compute scheduler, processes results, and
        determines the next batch of work.
        """
        logger.info("Runtime engine execution loop started.")

        # The work queue is populated in start()

        while self._is_running:
            self._pause_event.wait()  # This will block if the event is cleared (paused)
            if not self._is_running:  # Re-check after pause
                break

            # --- Build a batch of work from the queue ---
            current_batch = []
            while not self.work_queue.empty() and len(current_batch) < self.compute_scheduler.max_workers:
                try:
                    # Get a work unit, ignoring priority and counter
                    _, _, work_unit = self.work_queue.get_nowait()

                    # Discard work for cancelled trials
                    if work_unit.trial_id in self._cancelled_trials:
                        logger.info(f"Discarding cancelled work unit for trial {work_unit.trial_id}")
                        continue

                    current_batch.append(work_unit)
                except queue.Empty:
                    break  # Should not happen due to the while condition, but for safety

            if not current_batch:
                if all(
                    t.status in (TrialStatus.COMPLETED, TrialStatus.PRUNED)
                    for t in self.datastore.get_all_trials().values()
                ):
                    logger.info("All trials are completed or pruned. Shutting down.")
                    break
                threading.Event().wait(0.5)
                continue

            # --- Execute the batch and process results ---
            results_iterator = self.compute_scheduler.run(current_batch)
            for work_unit, result in results_iterator:
                self._pause_event.wait()
                if not self._is_running:
                    break
                self._process_completed_work_unit(work_unit, result)

        self._is_running = False
        logger.info("Runtime engine execution loop finished.")

    def _process_completed_work_unit(self, work_unit: WorkUnit, result: dict):
        """Handles the result of a single completed work unit.

        This involves updating the datastore, calling callbacks, analyzing for
        insights, and scheduling the next units of work.
        """
        # Get the trial's status *before* the update to detect transitions.
        original_status = self.datastore.get_trial(work_unit.trial_id).status
        updated_trial = self.datastore.record_work_unit_result(work_unit, result)

        if not updated_trial:
            logger.warning(
                f"Could not find trial {work_unit.trial_id} to record result."
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
            # No further work will be scheduled for this trial.
        else:
            # --- Insight Generation ---
            # We need to know if the trial just finished to decide which analyses to run.
            is_newly_finished = (
                updated_trial.status in (TrialStatus.COMPLETED, TrialStatus.PRUNED)
                and original_status
                not in (TrialStatus.COMPLETED, TrialStatus.PRUNED)
            )

            # Always run the lightweight, per-epoch analysis.
            insights = self.insight_engine.analyze_on_epoch(updated_trial)

            # If the trial just finished, also run the expensive, summary analysis.
            if is_newly_finished:
                logger.info(
                    f"Trial {updated_trial.id} has finished. Running final analysis."
                )
                insights.extend(self.insight_engine.analyze_on_finish(updated_trial))

            if insights and self.insights_callback:
                self.insights_callback([insight.__dict__ for insight in insights])

            # --- Scheduling Next Work ---
            # Only schedule next work if the trial is still active
            if updated_trial.status == TrialStatus.ACTIVE:
                next_work_units = self.adaptive_scheduler.get_next_work_units(
                    updated_trial, self.datastore.get_all_trials()
                )
                for next_wu in next_work_units:
                    self._put_work_in_queue(next_wu)

        # --- UI Callback ---
        # Always call the UI callback to update the trial's status, even on failure.
        if self.trial_updated_callback:
            self.trial_updated_callback(updated_trial.to_dict())

    def submit_work(
        self, work_units: List[WorkUnit]
    ) -> Iterator[Tuple[WorkUnit, dict]]:
        """Submits a list of work units to the scheduler and yields results.

        Note: This is a lower-level API that bypasses the adaptive scheduler.
        It's intended for specific use cases, not general experiment execution.
        """
        if not self._is_running:
            raise RuntimeError("Runtime Engine is not running.")
        return self.compute_scheduler.run(work_units)
