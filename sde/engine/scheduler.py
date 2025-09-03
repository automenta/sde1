import concurrent.futures
import time
import uuid
import traceback
from typing import List, Callable, Dict
import dataclasses
import threading
import logging

from sde.core.types import Trial, WorkUnit, TrialStatus, WorkUnitType
from sde.engine.worker import Worker
from sde.challenges import AVAILABLE_DATASETS
from sde.models import AVAILABLE_MODELS
from sde.exploration.schedulers import AdaptiveScheduler
from sde.engine.insight import InsightEngine

# Use Python's logging module for backend logs
logger = logging.getLogger(__name__)

class Signal:
    """A simple signal implementation to remove Qt dependency from the core engine."""
    def __init__(self, *arg_types):
        self._callbacks: List[Callable] = []

    def connect(self, callback: Callable):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.error(f"Error in signal callback: {traceback.format_exc()}")

def execute_work_unit_in_process(
    work_unit: WorkUnit,
    trial: Trial,
    model_name: str,
    dataset_name: str,
    enable_checkpointing: bool,
    checkpoints_dir: str
) -> dict:
    """
    A wrapper function that initializes a Worker in a new process
    and executes the given WorkUnit. It's a top-level function to be pickleable.
    """
    try:
        model_def = AVAILABLE_MODELS[model_name]
        dataset_def = AVAILABLE_DATASETS[dataset_name]
        worker = Worker(
            model_def=model_def,
            dataset_def=dataset_def,
            checkpoints_dir=checkpoints_dir
        )
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
    trial_profiled = Signal(str, float)
    experiment_finished = Signal()
    insight_generated = Signal(str)

    def __init__(self, trials: List[Trial], dataset_name: str, adaptive_scheduler: AdaptiveScheduler, max_workers: int = 2, enable_checkpointing: bool = False, checkpoints_dir: str = './checkpoints'):
        self.trials = {t.id: t for t in trials}
        self.dataset_name = dataset_name
        self.adaptive_scheduler = adaptive_scheduler
        self.max_workers = max_workers
        self.enable_checkpointing = enable_checkpointing
        self.checkpoints_dir = checkpoints_dir
        self.insight_engine = InsightEngine(
            self.trials,
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing
        )
        self.work_queue: List[WorkUnit] = []
        self.executor = None
        self.running_futures: Dict[concurrent.futures.Future, WorkUnit] = {}

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
        """Initializes the scheduler and starts the main execution loop."""
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.log_message.emit("INFO: Scheduler starting...")

        try:
            self._run_profiling_phase()
            self.log_message.emit("INFO: Profiling complete. Starting main experiment.")
            self.work_queue = self.adaptive_scheduler.get_initial_work_units(self.trials)
            for trial in self.trials.values():
                self.trial_updated.emit(trial.to_dict())
            self._main_loop()
        finally:
            self.shutdown()

    def _main_loop(self):
        """The core, simplified execution loop of the scheduler."""
        while self._is_running:
            if not self.running_futures and not self.work_queue:
                self.log_message.emit("INFO: All work is complete.")
                break

            with self.lock:
                is_paused = self._is_paused
                active_workers = self._active_workers

            if is_paused:
                time.sleep(0.5)
                continue

            self._dispatch_work(active_workers)
            self._process_completed_work()
            time.sleep(0.1) # Short sleep to prevent busy-waiting

    def _dispatch_work(self, active_workers: int):
        """Dispatches new work from the queue if workers are available."""
        while self.work_queue and len(self.running_futures) < active_workers:
            work_unit = self.work_queue.pop(0)
            trial = self.trials.get(work_unit.trial_id)

            if not trial or trial.status != TrialStatus.ACTIVE:
                self.log_message.emit(f"WARN: Skipping work for trial {work_unit.trial_id} with status {trial.status if trial else 'UNKNOWN'}.")
                continue

            future = self.executor.submit(
                execute_work_unit_in_process,
                work_unit, trial, trial.algorithm_name,
                self.dataset_name, self.enable_checkpointing, self.checkpoints_dir
            )
            self.running_futures[future] = work_unit
            log_epoch = f", Epoch {trial.current_epoch + 1}" if work_unit.type == WorkUnitType.TRAIN_EPOCH else ""
            self.log_message.emit(f"INFO: Dispatched: {work_unit.type.value} for Trial {trial.id}{log_epoch}")

    def _process_completed_work(self):
        """Processes any futures that have completed."""
        if not self.running_futures:
            return

        done_futures, _ = concurrent.futures.wait(
            self.running_futures.keys(),
            timeout=0, # Non-blocking
            return_when=concurrent.futures.FIRST_COMPLETED
        )

        for future in done_futures:
            work_unit = self.running_futures.pop(future)
            try:
                result = future.result()
                if 'error' in result:
                    self._handle_work_error(work_unit, result['error'])
                else:
                    self._process_result(work_unit, result)
            except Exception as e:
                self._handle_work_error(work_unit, traceback.format_exc())
            finally:
                self.trial_updated.emit(self.trials[work_unit.trial_id].to_dict())

    def _handle_work_error(self, work_unit: WorkUnit, error_traceback: str):
        """Handles a failed work unit, logging the error and pruning the trial."""
        self.log_message.emit(f"ERROR: Trial {work_unit.trial_id} failed in worker process.")
        self.log_message.emit(f"--- TRACEBACK ---\n{error_traceback}\n---")
        self.trials[work_unit.trial_id].status = TrialStatus.PRUNED

    def _run_profiling_phase(self):
        """Runs profiling for each unique algorithm to estimate epoch time."""
        self.log_message.emit("INFO: Starting performance profiling phase...")
        unique_algorithms = {t.algorithm_name: t for t in self.trials.values()}
        if not unique_algorithms:
            return

        profiling_work = [WorkUnit(trial_id=t.id, type=WorkUnitType.PROFILE_SPEED) for t in unique_algorithms.values()]
        self.work_queue.extend(profiling_work)

        while any(w.type == WorkUnitType.PROFILE_SPEED for w in self.work_queue) or \
              any(self.running_futures.get(f).type == WorkUnitType.PROFILE_SPEED for f in self.running_futures if f.running()):
            self._dispatch_work(self.max_workers)
            self._process_completed_work()
            time.sleep(0.1)

    def _process_result(self, work_unit: WorkUnit, result: dict):
        """Routes a result to the appropriate handler based on WorkUnitType."""
        if work_unit.type == WorkUnitType.PROFILE_SPEED:
            self._process_profiling_result(work_unit, result)
        elif work_unit.type == WorkUnitType.TRAIN_EPOCH:
            self._process_training_result(work_unit, result)

    def _process_profiling_result(self, work_unit: WorkUnit, result: dict):
        """Processes the result of a PROFILE_SPEED work unit."""
        trial = self.trials[work_unit.trial_id]
        est_time = result.get('profile_results', {}).get('est_time_per_epoch')
        if est_time is not None:
            self.log_message.emit(f"INFO: Profiled {trial.algorithm_name}: {est_time:.2f}s/epoch")
            for t in self.trials.values():
                if t.algorithm_name == trial.algorithm_name:
                    t.est_time_per_epoch = est_time
                    self.trial_profiled.emit(t.id, est_time)

    def _process_training_result(self, work_unit: WorkUnit, result: dict):
        """Processes the result of a TRAIN_EPOCH work unit."""
        trial = self.trials[work_unit.trial_id]

        # 1. Update trial state and metrics
        state_updates = result['state_updates']
        trial.current_epoch = state_updates.get('current_epoch', trial.current_epoch)
        if state_updates.get('checkpoint_path') is not None:
            trial.checkpoint_path = state_updates['checkpoint_path']

        for name, value in result.get('metrics', {}).items():
            trial.results.setdefault(name, []).append((trial.current_epoch, value))
        self.log_message.emit(f"Result for Trial {trial.id}, Epoch {trial.current_epoch}: {result.get('metrics', {})}")
        self.trial_updated.emit(trial.to_dict())

        # 2. Analyze for insights
        insights = self.insight_engine.analyze(trial)
        for insight in insights:
            self.insight_generated.emit(insight)

        # 3. Get next work from the adaptive scheduler
        new_work = self.adaptive_scheduler.get_next_work_units(trial, self.trials)
        self.work_queue.extend(new_work)

        # 4. Emit updates for any newly pruned trials
        for t in self.trials.values():
            if t.status == TrialStatus.PRUNED and t.id != trial.id:
                self.trial_updated.emit(t.to_dict())

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
    from sde.exploration.schedulers import SuccessiveHalvingScheduler
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    trials_to_run = [
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="SimpleCNN",
            hyperparameters={
                'model_params': {'dropout_rate': 0.25},
                'optimizer_params': {'name': 'Adam', 'lr': 0.01}
            }
        ),
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="LogisticRegression",
            hyperparameters={
                'optimizer_params': {'name': 'SGD', 'lr': 0.001}
            }
        ),
    ]

    DATASET = "MNIST"
    logger.info(f"Starting experiment with {len(trials_to_run)} trials on {DATASET}.")

    adaptive_scheduler = SuccessiveHalvingScheduler(metric="accuracy", increasing=True)

    scheduler = Scheduler(
        trials=trials_to_run,
        dataset_name=DATASET,
        adaptive_scheduler=adaptive_scheduler,
        max_workers=2,
        enable_checkpointing=True
    )

    # Example of connecting a signal
    scheduler.log_message.connect(lambda msg: logger.info(f"SCHEDULER: {msg}"))
    scheduler.insight_generated.connect(lambda msg: logger.info(f"INSIGHT: {msg}"))

    start_time = time.time()
    scheduler.start()
    end_time = time.time()

    logger.info(f"\n--- Experiment Finished in {end_time - start_time:.2f} seconds ---")
    for trial in scheduler.trials.values():
        results_str = "".join([f"\n    {m}: {v}" for m, v in trial.results.items()])
        logger.info(
            f"\nTrial ID: {trial.id}"
            f"\n  Algorithm: {trial.algorithm_name}"
            f"\n  Status: {trial.status.value}"
            f"\n  Hyperparameters: {trial.hyperparameters}"
            f"\n  Results: {results_str}"
        )
