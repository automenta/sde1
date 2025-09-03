import concurrent.futures
import time
import uuid
import traceback
from typing import List, Callable
import dataclasses
import threading

from sde.core.types import Trial, WorkUnit, TrialStatus
from sde.engine.datastore import DataStore
from sde.engine.worker import Worker
from sde.challenges import AVAILABLE_DATASETS
from sde.models import AVAILABLE_MODELS
from sde.exploration.schedulers import AdaptiveScheduler, SuccessiveHalvingScheduler
from sde.engine.insight import InsightEngine

class Signal:
    """A simple signal implementation to remove Qt dependency from the core engine."""
    def __init__(self, *arg_types):
        self._callbacks: List[Callable] = []

    def connect(self, callback: Callable):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)

def trial_to_dict(trial: Trial) -> dict:
    """Converts a Trial dataclass instance to a dictionary for signal emission."""
    d = dataclasses.asdict(trial)
    d['status'] = trial.status.value
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
    print(f"[Worker Process] Starting work for Trial {work_unit.trial_id[:6]}")
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
    """
    log_message = Signal(str)
    trial_updated = Signal(dict)
    experiment_finished = Signal()
    insight_generated = Signal(str)

    def __init__(self, trials: List[Trial], dataset_name: str, adaptive_scheduler: AdaptiveScheduler, max_workers: int = 2, enable_checkpointing: bool = False):
        self.datastore = DataStore(trials)
        self.dataset_name = dataset_name
        self.adaptive_scheduler = adaptive_scheduler
        self.max_workers = max_workers
        self.enable_checkpointing = enable_checkpointing
        self.insight_engine = InsightEngine(
            self.datastore,
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing
        )
        self.work_queue: List[WorkUnit] = []
        self.executor = None
        self.running_futures = {}

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
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.log_message.emit("INFO: Scheduler starting with adaptive policy...")
        try:
            self.work_queue = self.adaptive_scheduler.get_initial_work_units(self.datastore)
            for trial in self.datastore.get_all_trials():
                self.trial_updated.emit(trial_to_dict(trial))
            self._main_loop()
        finally:
            self.shutdown()

    def _main_loop(self):
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
            time.sleep(0.1)

    def _handle_paused_state(self):
        self._process_completed_futures()
        time.sleep(0.5)

    def _dispatch_work_units(self, active_workers: int):
        while self.work_queue and len(self.running_futures) < active_workers:
            work_unit = self.work_queue.pop(0)
            trial = self.datastore.get_trial(work_unit.trial_id)
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
        if not self.running_futures:
            return
        done_futures, _ = concurrent.futures.wait(
            self.running_futures.keys(),
            timeout=0.1,
            return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done_futures:
            work_unit = self.running_futures.pop(future)
            trial_id = work_unit.trial_id
            try:
                result = future.result()
                if 'error' in result:
                    error_traceback = result['error']
                    self.log_message.emit(f"ERROR: Trial {trial_id} failed in worker process.")
                    self.log_message.emit(f"--- TRACEBACK ---\n{error_traceback}\n---")
                    self.datastore.set_trial_status(trial_id, TrialStatus.PRUNED)
                else:
                    self._process_result(work_unit, result)
            except Exception as e:
                self.log_message.emit(f"ERROR: Scheduler failed to process result for trial {trial_id}: {e}")
                self.log_message.emit(f"--- TRACEBACK ---\n{traceback.format_exc()}\n---")
                self.datastore.set_trial_status(trial_id, TrialStatus.PRUNED)
            finally:
                self.trial_updated.emit(trial_to_dict(self.datastore.get_trial(trial_id)))

    def _process_result(self, work_unit: WorkUnit, result: dict):
        trial_id = work_unit.trial_id
        self.datastore.update_trial_from_result(trial_id, result)
        trial = self.datastore.get_trial(trial_id)
        self.log_message.emit(f"Result for Trial {trial.id} ({trial.algorithm_name}), Epoch {trial.current_epoch}: {result.get('metrics', {})}")
        self.trial_updated.emit(trial_to_dict(trial))

        insights = self.insight_engine.analyze(trial)
        for insight in insights:
            self.insight_generated.emit(f"[INSIGHT] {insight.message}")

        new_work_units = self.adaptive_scheduler.get_next_work_units(trial, self.datastore)
        if new_work_units:
            self.work_queue.extend(new_work_units)

        # The adaptive scheduler may have pruned trials, so we need to update their status in the UI
        for t in self.datastore.get_trials_by_status(TrialStatus.PRUNED):
             self.trial_updated.emit(trial_to_dict(t))

    def stop(self):
        if self._is_running:
            self.log_message.emit("INFO: Shutdown signal received...")
            self._is_running = False

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        self.log_message.emit("Scheduler shutdown complete.")
        self.experiment_finished.emit()

if __name__ == "__main__":
    trials_to_run = [
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="SimpleCNN",
            hyperparameters={
                'model_params': {'dropout_rate': 0.25},
                'optimizer_params': {'lr': 0.01},
                'loader_params': {'batch_size': 128}
            }
        ),
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="LogisticRegression",
            hyperparameters={
                'optimizer_params': {'lr': 0.001},
                'loader_params': {'batch_size': 256}
            }
        ),
    ]
    DATASET = "MNIST"
    print(f"Starting experiment with {len(trials_to_run)} trials on {DATASET}.")
    adaptive_scheduler = SuccessiveHalvingScheduler(metric="accuracy", increasing=True)
    scheduler = Scheduler(
        trials=trials_to_run,
        dataset_name=DATASET,
        adaptive_scheduler=adaptive_scheduler,
        max_workers=2,
        enable_checkpointing=True
    )
    start_time = time.time()
    scheduler.start()
    end_time = time.time()
    print(f"\n--- Experiment Finished in {end_time - start_time:.2f} seconds ---")
    for trial in scheduler.datastore.get_all_trials():
        print(f"\nTrial ID: {trial.id}")
        print(f"  Algorithm: {trial.algorithm_name}")
        print(f"  Status: {trial.status.value}")
        print(f"  Hyperparameters: {trial.hyperparameters}")
        print(f"  Results:")
        for metric, values in trial.results.items():
            print(f"    {metric}: {values}")
