import concurrent.futures
import time
import uuid
import traceback
from typing import List, Callable
import dataclasses
import threading

from sde.core.types import Trial, WorkUnit, TrialStatus
from sde.engine.worker import Worker
from sde.challenges import AVAILABLE_DATASETS
from sde.models import AVAILABLE_MODELS
from sde.exploration.schedulers import AdaptiveScheduler
from sde.engine.insight import InsightEngine

class Signal:
    """A simple signal implementation to remove Qt dependency from the core engine."""
    def __init__(self, *arg_types):
        self._callbacks: List[Callable] = []

    def connect(self, callback: Callable):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            # In a real-world scenario, you might want to handle exceptions here
            callback(*args, **kwargs)

def trial_to_dict(trial: Trial) -> dict:
    """Converts a Trial dataclass instance to a dictionary for signal emission."""
    d = dataclasses.asdict(trial)
    d['status'] = trial.status.value # Enums need to be converted to string values
    return d

def execute_work_unit_in_process(
    work_unit: WorkUnit,
    trial: Trial,
    model_name: str,
    dataset_name: str,
    enable_checkpointing: bool
) -> dict:
    """
    A wrapper function that initializes a Worker in a new process
    and executes the given WorkUnit. It's a top-level function to be pickleable.
    """
    try:
        model_def = AVAILABLE_MODELS[model_name]
        dataset_def = AVAILABLE_DATASETS[dataset_name]
        worker = Worker(model_def=model_def, dataset_def=dataset_def)
        return worker.execute_work_unit(work_unit, trial, enable_checkpointing)
    except Exception:
        return {
            'error': traceback.format_exc()
        }


class Scheduler:
    """
    Manages the lifecycle of an experiment, dispatching WorkUnits to a
    pool of workers based on decisions from an AdaptiveScheduler.
    This class is designed to be framework-agnostic.
    """
    log_message = Signal(str)
    trial_updated = Signal(dict)
    experiment_finished = Signal()
    insight_generated = Signal(str)

    def __init__(self, trials: List[Trial], dataset_name: str, adaptive_scheduler: AdaptiveScheduler, max_workers: int = 2, enable_checkpointing: bool = False):
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
        self.lock = threading.Lock()
        self._is_running = True
        self._is_paused = False
        self._active_workers = max_workers

    def pause(self):
        with self.lock:
            if not self._is_paused:
                self.log_message.emit("INFO: Pausing experiment. Finishing active work...")
                self._is_paused = True

    def resume(self):
        with self.lock:
            if self._is_paused:
                self.log_message.emit("INFO: Resuming experiment...")
                self._is_paused = False

    def set_throttle(self, percentage: int):
        if not (1 <= percentage <= 100):
            self.log_message.emit(f"WARN: Throttle percentage must be between 1-100. Got {percentage}.")
            return

        with self.lock:
            new_worker_count = max(1, int(self.max_workers * (percentage / 100.0)))
            if new_worker_count != self._active_workers:
                self._active_workers = new_worker_count
                self.log_message.emit(f"INFO: Throttle set to {percentage}%. Active workers: {self._active_workers}/{self.max_workers}")

    def start(self):
        """
        Initializes the scheduler and starts the main execution loop.
        """
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.log_message.emit("INFO: Scheduler starting with adaptive policy...")

        try:
            # Populate the initial work queue
            self.work_queue = self.adaptive_scheduler.get_initial_work_units(self.trials)
            for trial in self.trials.values():
                self.trial_updated.emit(trial_to_dict(trial))

            # Start the main loop
            self._main_loop()
        finally:
            # Ensure shutdown is always called
            self.shutdown()

    def _main_loop(self):
        """
        The core execution loop of the scheduler.
        """
        while self._is_running:
            with self.lock:
                is_paused = self._is_paused
                active_workers = self._active_workers

            if is_paused:
                self._handle_paused_state()
                continue

            self._dispatch_work_units(active_workers)
            self._process_completed_futures()

            if not self.running_futures and not self.work_queue:
                self.log_message.emit("INFO: All work is complete.")
                break

            time.sleep(0.1) # Prevent busy-waiting

    def _handle_paused_state(self):
        """Processes completed work while paused without dispatching new work."""
        self._process_completed_futures()
        time.sleep(0.5) # Sleep longer when paused

    def _dispatch_work_units(self, active_workers: int):
        """Dispatches new work from the queue if workers are available."""
        while self.work_queue and len(self.running_futures) < active_workers:
            work_unit = self.work_queue.pop(0)
            trial = self.trials[work_unit.trial_id]

            # Ensure we don't schedule work for a trial that's no longer active
            if trial.status != TrialStatus.ACTIVE:
                self.log_message.emit(f"WARN: Skipping work for trial {trial.id} with status {trial.status.value}.")
                continue

            future = self.executor.submit(
                execute_work_unit_in_process,
                work_unit, trial, trial.algorithm_name,
                self.dataset_name, self.enable_checkpointing
            )
            self.running_futures[future] = work_unit
            self.log_message.emit(f"INFO: Dispatched: Trial {trial.id} ({trial.algorithm_name}), Epoch {trial.current_epoch + 1}")

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
                # Check if the worker process returned an error
                if 'error' in result:
                    error_traceback = result['error']
                    self.log_message.emit(f"ERROR: Trial {work_unit.trial_id} failed in worker process.")
                    self.log_message.emit(f"--- TRACEBACK ---\n{error_traceback}\n---")
                    self.trials[work_unit.trial_id].status = TrialStatus.PRUNED
                else:
                    self._process_result(work_unit, result)
            except Exception as e:
                # This catches errors in the scheduler logic itself (e.g., _process_result)
                self.log_message.emit(f"ERROR: Scheduler failed to process result for trial {work_unit.trial_id}: {e}")
                self.log_message.emit(f"--- TRACEBACK ---\n{traceback.format_exc()}\n---")
                self.trials[work_unit.trial_id].status = TrialStatus.PRUNED
            finally:
                # Ensure trial UI is always updated, even on failure
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
        if self._is_running:
            self.log_message.emit("INFO: Shutdown signal received. The scheduler will stop after finishing active work units.")
            self._is_running = False

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        self.log_message.emit("Scheduler shutdown complete.")
        self.experiment_finished.emit()


# A simple test block to run the backend from the command line
if __name__ == "__main__":
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

    adaptive_scheduler = SuccessiveHalvingScheduler(metric="accuracy", increasing=True)

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
