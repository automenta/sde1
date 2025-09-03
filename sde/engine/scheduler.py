import concurrent.futures
import uuid
import traceback
from typing import List, Dict, Tuple, Iterator
import logging

from sde.core.types import Trial, WorkUnit, TrialStatus
from sde.challenges import AVAILABLE_DATASETS
from sde.models import AVAILABLE_MODELS
from sde.engine.datastore import DataStore

logger = logging.getLogger(__name__)

def execute_work_unit_in_process(
    work_unit: WorkUnit,
    trial: Trial,
    model_name: str,
    dataset_name: str,
    enable_checkpointing: bool,
    checkpoints_dir: str
) -> Tuple[WorkUnit, dict]:
    """
    A wrapper function that initializes a Worker in a new process, executes
    the given WorkUnit, and returns the result along with the original WorkUnit.
    It's a top-level function to be pickleable.
    """
    try:
        from sde.engine.worker import Worker # Import inside for pickling
        model_def = AVAILABLE_MODELS[model_name]
        dataset_def = AVAILABLE_DATASETS[dataset_name]
        worker = Worker(
            model_def=model_def,
            dataset_def=dataset_def,
            checkpoints_dir=checkpoints_dir
        )
        result = worker.execute_work_unit(work_unit, trial, enable_checkpointing)
        return work_unit, result
    except Exception:
        return work_unit, {'error': traceback.format_exc()}


class Scheduler:
    """
    Manages a pool of worker processes using concurrent.futures. It submits
    work units and yields results as they are completed, using the
    `as_completed` pattern for efficiency.
    """

    def __init__(
        self,
        datastore: DataStore,
        dataset_name: str,
        max_workers: int = 2,
        enable_checkpointing: bool = False,
        checkpoints_dir: str = './checkpoints'
    ):
        self.datastore = datastore
        self.dataset_name = dataset_name
        self.max_workers = max_workers
        self.enable_checkpointing = enable_checkpointing
        self.checkpoints_dir = checkpoints_dir
        self.executor = None
        self._is_running = False

    def start(self):
        """Initializes the process pool executor."""
        if not self._is_running:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
            self._is_running = True

    def stop(self):
        """Stops the scheduler and shuts down the executor."""
        if self._is_running:
            self._is_running = False
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def run(self, work_units: List[WorkUnit]) -> Iterator[Tuple[WorkUnit, dict]]:
        """
        Submits work units to the executor and yields results as they complete.
        This is a generator function.
        """
        if not self._is_running or not self.executor:
            raise RuntimeError("Scheduler must be started before running work.")

        futures = {}
        for work_unit in work_units:
            trial = self.datastore.get_trial(work_unit.trial_id)
            if not trial or trial.status != TrialStatus.ACTIVE:
                continue

            future = self.executor.submit(
                execute_work_unit_in_process,
                work_unit, trial, trial.algorithm_name,
                self.dataset_name, self.enable_checkpointing, self.checkpoints_dir
            )
            futures[future] = work_unit

        for future in concurrent.futures.as_completed(futures):
            # If the scheduler was stopped while waiting, exit the loop.
            if not self._is_running:
                break

            try:
                # The result from execute_work_unit_in_process is (WorkUnit, dict)
                work_unit, result = future.result()
                yield work_unit, result
            except concurrent.futures.CancelledError:
                # This can happen during shutdown.
                logger.warning("A worker future was cancelled.")
                continue
            except Exception:
                # Should be caught by the wrapper, but as a fallback:
                work_unit = futures[future]
                yield work_unit, {'error': traceback.format_exc()}
