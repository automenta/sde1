import concurrent.futures
import time
import uuid
from typing import List
import dataclasses

from PyQt6.QtCore import QObject, pyqtSignal

from sde.core.types import Trial, WorkUnit, WorkUnitType, TrialStatus
from sde.engine.worker import Worker
from sde.challenges import mnist
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
def execute_work_unit_in_process(work_unit: WorkUnit, trial: Trial) -> dict:
    """
    A wrapper function that initializes a Worker in a new process
    and executes the given WorkUnit.
    """
    # Each process creates its own worker instance.
    worker = Worker(challenge_module=mnist)
    return worker.execute_work_unit(work_unit, trial)


class Scheduler(QObject):
    """
    Manages the lifecycle of an experiment, dispatching WorkUnits to a
    pool of workers based on decisions from an AdaptiveScheduler.
    """
    log_message = pyqtSignal(str)
    trial_updated = pyqtSignal(dict)
    experiment_finished = pyqtSignal()
    insight_generated = pyqtSignal(str)

    def __init__(self, trials: List[Trial], adaptive_scheduler: AdaptiveScheduler, max_workers: int = 2):
        super().__init__()
        self.trials = {t.id: t for t in trials}
        self.adaptive_scheduler = adaptive_scheduler
        self.max_workers = max_workers
        self.insight_engine = InsightEngine(
            self.trials,
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing
        )
        self.work_queue: List[WorkUnit] = []
        self.executor = None
        self.running_futures = {}
        self._is_running = True

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
            # 2. Dispatch work from the queue
            while self.work_queue and len(self.running_futures) < self.max_workers:
                work_unit = self.work_queue.pop(0)
                trial = self.trials[work_unit.trial_id]

                future = self.executor.submit(execute_work_unit_in_process, work_unit, trial)
                self.running_futures[future] = work_unit
                self.log_message.emit(f"Dispatched: Trial {trial.id}, Epoch {trial.current_epoch + 1}")

            # 3. Process completed futures
            if not self.running_futures:
                # If no work is running and queue is empty, we might be done
                if not self.work_queue:
                    self.log_message.emit("No more work to schedule.")
                    break
                else: # Still work in queue, but all workers are free, no need to wait
                    continue

            done_futures, _ = concurrent.futures.wait(
                self.running_futures.keys(),
                timeout=0.5,
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

        self.shutdown()

    def _process_result(self, work_unit: WorkUnit, result: dict):
        trial = self.trials[work_unit.trial_id]

        # Update trial state from result
        state_updates = result['state_updates']
        trial.current_epoch = state_updates['current_epoch']
        trial.checkpoint_path = state_updates['checkpoint_path']

        metrics = result['metrics']
        for metric_name, value in metrics.items():
            if metric_name not in trial.results:
                trial.results[metric_name] = []
            trial.results[metric_name].append((trial.current_epoch, value))

        self.log_message.emit(f"Result for Trial {trial.id}, Epoch {trial.current_epoch}: {metrics}")
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
            algorithm_name="SimpleCNN",
            hyperparameters={
                'model_params': {'dropout_rate': 0.5},
                'optimizer_params': {'lr': 0.001}
            }
        ),
    ]

    MAX_EPOCHS = 1

    print(f"Starting experiment with {len(trials_to_run)} trials for {MAX_EPOCHS} epochs.")

    scheduler = Scheduler(trials=trials_to_run, max_epochs=MAX_EPOCHS, max_workers=2)

    start_time = time.time()
    scheduler.start()
    end_time = time.time()

    print(f"\n--- Experiment Finished in {end_time - start_time:.2f} seconds ---")
    for trial in scheduler.trials.values():
        print(f"\nTrial ID: {trial.id}")
        print(f"  Status: {trial.status.value}")
        print(f"  Hyperparameters: {trial.hyperparameters}")
        print(f"  Results:")
        for metric, values in trial.results.items():
            print(f"    {metric}: {values}")
