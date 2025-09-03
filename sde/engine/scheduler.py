import concurrent.futures
import time
import uuid
from typing import List
import dataclasses

from PyQt6.QtCore import QObject, pyqtSignal

from sde.core.types import Trial, WorkUnit, WorkUnitType, TrialStatus
from sde.engine.worker import Worker
from sde.challenges import AVAILABLE_DATASETS
from sde.models import AVAILABLE_MODELS
from sde.exploration.schedulers import AdaptiveScheduler
from sde.engine.insight import InsightEngine

def trial_to_dict(trial: Trial) -> dict:
    """Converts a Trial dataclass instance to a dictionary for signal emission."""
    # Enums are not directly JSON serializable, so convert them to their string values
    d = dataclasses.asdict(trial)
    d['status'] = trial.status.value
    return d

# This top-level function will be sent to the ProcessPoolExecutor processes.
# It needs to be defined at the top level of the module to be pickleable.
def execute_work_unit_in_process(work_unit: WorkUnit, trial: Trial, model_name: str, dataset_name: str, enable_checkpointing: bool) -> dict:
    """
    A wrapper function that initializes a Worker in a new process
    and executes the given WorkUnit.
    """
    # Each process looks up the definitions from the registries
    model_def = AVAILABLE_MODELS[model_name]
    dataset_def = AVAILABLE_DATASETS[dataset_name]

    # Each process creates its own worker instance for the specific model/dataset pair.
    worker = Worker(model_def=model_def, dataset_def=dataset_def)
    return worker.execute_work_unit(work_unit, trial, enable_checkpointing)


class Scheduler(QObject):
    """
    Manages the lifecycle of an experiment, dispatching WorkUnits to a
    pool of workers based on decisions from an AdaptiveScheduler.
    """
    log_message = pyqtSignal(str)
    trial_updated = pyqtSignal(dict)
    experiment_finished = pyqtSignal()
    insight_generated = pyqtSignal(str)

    def __init__(self, trials: List[Trial], dataset_name: str, adaptive_scheduler: AdaptiveScheduler, max_workers: int = 2, enable_checkpointing: bool = False):
        super().__init__()
        self.trials = {t.id: t for t in trials}
        self.dataset_name = dataset_name
        self.adaptive_scheduler = adaptive_scheduler
        self.max_workers = max_workers
        self.enable_checkpointing = enable_checkpointing
        self.insight_engine = InsightEngine(
            self.trials,
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing
        )
        self.work_queue: List[WorkUnit] = []
        self.executor = None
        self.running_futures = {}

        # --- State flags for execution control ---
        self._is_running = True
        self._is_paused = False
        self._active_workers = max_workers

    def pause(self):
        self.log_message.emit("INFO: Pausing experiment. Finishing active work...")
        self._is_paused = True

    def resume(self):
        self.log_message.emit("INFO: Resuming experiment...")
        self._is_paused = False

    def set_throttle(self, percentage: int):
        if not (0 < percentage <= 100):
            self.log_message.emit(f"WARN: Throttle percentage must be between 1-100. Got {percentage}.")
            return

        new_worker_count = max(1, int(self.max_workers * (percentage / 100.0)))
        if new_worker_count != self._active_workers:
            self._active_workers = new_worker_count
            self.log_message.emit(f"INFO: Throttle set to {percentage}%. Active workers: {self._active_workers}/{self.max_workers}")

    def start(self):
        """
        Starts the main scheduling loop.
        """
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.log_message.emit("Scheduler starting with adaptive policy...")

        # 1. Get initial work from the adaptive scheduler
        self.work_queue = self.adaptive_scheduler.get_initial_work_units(self.trials)
        for trial in self.trials.values():
             # Emit the initial state of all trials
            self.trial_updated.emit(trial_to_dict(trial))

        while self._is_running:
            # If paused, sleep and check for completed work without dispatching new work
            if self._is_paused:
                self._process_completed_futures()
                time.sleep(0.5)
                continue

            # 2. Dispatch work from the queue, respecting the throttle
            while self.work_queue and len(self.running_futures) < self._active_workers:
                work_unit = self.work_queue.pop(0)
                trial = self.trials[work_unit.trial_id]

                future = self.executor.submit(
                    execute_work_unit_in_process,
                    work_unit,
                    trial,
                    trial.algorithm_name,
                    self.dataset_name,
                    self.enable_checkpointing
                )
                self.running_futures[future] = work_unit
                self.log_message.emit(f"Dispatched: Trial {trial.id} ({trial.algorithm_name}), Epoch {trial.current_epoch + 1}")

            # 3. Process completed futures
            if not self.running_futures and not self.work_queue:
                self.log_message.emit("No more work to schedule.")
                break

            self._process_completed_futures()
            time.sleep(0.1) # Small sleep to prevent busy-waiting

        self.shutdown()

    def _process_completed_futures(self):
        """Helper to check for and process any finished work units."""
        if not self.running_futures:
            return

        done_futures, _ = concurrent.futures.wait(
            self.running_futures.keys(),
            timeout=0.1, # Short timeout to remain responsive
            return_when=concurrent.futures.FIRST_COMPLETED
        )

        for future in done_futures:
            work_unit = self.running_futures.pop(future)
            try:
                result = future.result()
                self._process_result(work_unit, result)
            except Exception as e:
                self.log_message.emit(f"ERROR: Work unit {work_unit} failed: {e}")
                self.trials[work_unit.trial_id].status = TrialStatus.PRUNED
                self.trial_updated.emit(trial_to_dict(self.trials[work_unit.trial_id]))

    def _process_result(self, work_unit: WorkUnit, result: dict):
        trial = self.trials[work_unit.trial_id]

        # Update trial state from result
        state_updates = result['state_updates']
        trial.current_epoch = state_updates['current_epoch']
        # Only update checkpoint path if a new one was created
        if state_updates['checkpoint_path'] is not None:
            trial.checkpoint_path = state_updates['checkpoint_path']

        metrics = result['metrics']
        for metric_name, value in metrics.items():
            if metric_name not in trial.results:
                trial.results[metric_name] = []
            trial.results[metric_name].append((trial.current_epoch, value))

        self.log_message.emit(f"Result for Trial {trial.id} ({trial.algorithm_name}), Epoch {trial.current_epoch}: {metrics}")
        self.trial_updated.emit(trial_to_dict(trial))

        # 3. Analyze for insights
        insights = self.insight_engine.analyze(trial)
        for insight in insights:
            self.insight_generated.emit(f"[INSIGHT] {insight.message}")

        # 4. Get next work units from the adaptive scheduler
        new_work_units = self.adaptive_scheduler.get_next_work_units(trial, self.trials)
        if new_work_units:
            self.work_queue.extend(new_work_units)

        # The adaptive scheduler may have pruned trials, so we need to update their status
        for t in self.trials.values():
            if t.status == TrialStatus.PRUNED:
                self.trial_updated.emit(trial_to_dict(t))

    def stop(self):
        self._is_running = False
        self.log_message.emit("Shutdown signal received. Finishing active work...")

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        self.log_message.emit("Scheduler shutdown complete.")
        self.experiment_finished.emit()


# A simple test block to run the backend from the command line
if __name__ == "__main__":
    from sde.exploration.schedulers import SuccessiveHalvingScheduler

    # Define a few sample trials with a structured hyperparameter format
    trials_to_run = [
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="SimpleCNN",
            hyperparameters={
                'model_params': {'dropout_rate': 0.25},
                'optimizer_params': {'lr': 0.01}
            }
        ),
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="LogisticRegression",
            hyperparameters={
                'optimizer_params': {'lr': 0.001}
            }
        ),
    ]

    # This test now runs on MNIST by default
    DATASET = "MNIST"
    print(f"Starting experiment with {len(trials_to_run)} trials on {DATASET}.")

    adaptive_scheduler = SuccessiveHalvingScheduler(metric="accuracy", increasing=True, max_rungs=4)

    scheduler = Scheduler(
        trials=trials_to_run,
        dataset_name=DATASET,
        adaptive_scheduler=adaptive_scheduler,
        max_workers=2,
        enable_checkpointing=True # Enabled for test
    )

    start_time = time.time()
    scheduler.start()
    end_time = time.time()

    print(f"\n--- Experiment Finished in {end_time - start_time:.2f} seconds ---")
    for trial in scheduler.trials.values():
        print(f"\nTrial ID: {trial.id}")
        print(f"  Algorithm: {trial.algorithm_name}")
        print(f"  Status: {trial.status.value}")
        print(f"  Hyperparameters: {trial.hyperparameters}")
        print(f"  Results:")
        for metric, values in trial.results.items():
            print(f"    {metric}: {values}")
