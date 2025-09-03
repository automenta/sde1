import time
import threading
import logging
import traceback
from typing import List, Callable

from sde.core.types import Trial, WorkUnit, TrialStatus, WorkUnitType
from sde.engine.datastore import DataStore
from sde.engine.scheduler import Scheduler
from sde.exploration.schedulers import AdaptiveScheduler
from sde.engine.insight import InsightEngine

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

class Orchestrator:
    """
    Manages the high-level lifecycle of an experiment, coordinating all other
    components like the DataStore, Scheduler, and InsightEngine.
    """
    log_message = Signal(str)
    trial_updated = Signal(dict)
    trial_profiled = Signal(str, float)
    experiment_finished = Signal()
    insight_generated = Signal(str)

    def __init__(
        self,
        trials: List[Trial],
        dataset_name: str,
        adaptive_scheduler: AdaptiveScheduler,
        max_workers: int = 2,
        enable_checkpointing: bool = False,
        checkpoints_dir: str = './checkpoints'
    ):
        self.datastore = DataStore(trials)
        self.adaptive_scheduler = adaptive_scheduler
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
        self._main_thread = None

    def start(self):
        """Initializes the scheduler and starts the main execution loop in a new thread."""
        if self._is_running:
            return

        self.log_message.emit("INFO: Orchestrator starting...")
        self._is_running = True
        self._main_thread = threading.Thread(target=self._run_experiment)
        self._main_thread.start()
        return self._main_thread

    def _run_experiment(self):
        try:
            self.scheduler.start()

            # Phase 1: Profiling
            self.log_message.emit("INFO: Starting performance profiling phase...")
            profiling_work = self._get_profiling_work()
            if profiling_work:
                for work_unit, result in self.scheduler.run(profiling_work):
                    if not self._is_running: break
                    self._process_profiling_result(work_unit, result)

            # Phase 2: Main Experiment Loop
            self.log_message.emit("INFO: Profiling complete. Starting main experiment.")
            work_queue = self.adaptive_scheduler.get_initial_work_units(self.datastore.get_all_trials())
            for trial in self.datastore.get_all_trials().values():
                self.trial_updated.emit(trial.to_dict())

            while self._is_running and work_queue:
                next_work_queue = []
                for work_unit, result in self.scheduler.run(work_queue):
                    if not self._is_running: break

                    if 'error' in result:
                        self._handle_work_error(work_unit, result['error'])
                    else:
                        new_work = self._process_training_result(work_unit, result)
                        next_work_queue.extend(new_work)

                    # Always emit an update for the trial that just finished
                    finished_trial = self.datastore.get_trial(work_unit.trial_id)
                    if finished_trial:
                        self.trial_updated.emit(finished_trial.to_dict())

                work_queue = next_work_queue

            if self._is_running:
                self.log_message.emit("INFO: All work is complete.")

        except Exception:
            self.log_message.emit(f"FATAL: An unexpected error occurred: {traceback.format_exc()}")
        finally:
            self.shutdown()

    def _get_profiling_work(self) -> List[WorkUnit]:
        """Gets the initial work units for the profiling phase."""
        trials = self.datastore.get_all_trials()
        unique_algorithms = {t.algorithm_name: t for t in trials.values()}
        return [WorkUnit(trial_id=t.id, type=WorkUnitType.PROFILE_SPEED) for t in unique_algorithms.values()]

    def _handle_work_error(self, work_unit: WorkUnit, error_traceback: str):
        self.log_message.emit(f"ERROR: Trial {work_unit.trial_id} failed.")
        self.log_message.emit(f"--- TRACEBACK ---\n{error_traceback}\n---")
        self.datastore.update_trial_status(work_unit.trial_id, TrialStatus.PRUNED)

    def _process_profiling_result(self, work_unit: WorkUnit, result: dict):
        trial = self.datastore.get_trial(work_unit.trial_id)
        est_time = result.get('profile_results', {}).get('est_time_per_epoch')
        if est_time is not None:
            self.log_message.emit(f"INFO: Profiled {trial.algorithm_name}: {est_time:.2f}s/epoch")
            self.datastore.update_algorithm_profile(trial.algorithm_name, est_time)
            for t in self.datastore.get_all_trials().values():
                if t.algorithm_name == trial.algorithm_name:
                    self.trial_profiled.emit(t.id, est_time)

    def _process_training_result(self, work_unit: WorkUnit, result: dict) -> List[WorkUnit]:
        """Processes a training result and returns the next batch of work."""
        # 1. Update datastore
        self.datastore.update_trial_state(work_unit, result)
        trial = self.datastore.get_trial(work_unit.trial_id)
        self.log_message.emit(f"Result for Trial {trial.id}, Epoch {trial.current_epoch}: {result.get('metrics', {})}")

        # 2. Analyze for insights
        insights = self.insight_engine.analyze(trial)
        for insight in insights:
            self.insight_generated.emit(insight)

        # 3. Get next work from the adaptive scheduler
        all_trials = self.datastore.get_all_trials()
        new_work = self.adaptive_scheduler.get_next_work_units(trial, all_trials)

        # 4. Emit updates for any newly pruned trials
        for t in all_trials.values():
            if t.status == TrialStatus.PRUNED:
                self.trial_updated.emit(t.to_dict())

        return new_work

    def stop(self):
        """Signals the main loop to stop and shuts down the scheduler."""
        if self._is_running:
            self.log_message.emit("INFO: Shutdown signal received.")
            self._is_running = False
            # The scheduler will be stopped in the finally block

    def shutdown(self):
        self.scheduler.stop()
        self.log_message.emit("Orchestrator shutdown complete.")
        self.experiment_finished.emit()

    def join(self):
        """Waits for the main experiment thread to complete."""
        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join()
