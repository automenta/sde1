import concurrent.futures
import multiprocessing
import threading
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
    checkpoints_dir: str,
) -> Tuple[WorkUnit, dict]:
    """
    A wrapper function that initializes a Worker in a new process, executes
    the given WorkUnit, and returns the result along with the original WorkUnit.
    It's a top-level function to be pickleable.
    """
    try:
        from sde.engine.worker import Worker  # Import inside for pickling

        model_def = AVAILABLE_MODELS[model_name]
        dataset_def = AVAILABLE_DATASETS[dataset_name]
        worker = Worker(
            model_def=model_def,
            dataset_def=dataset_def,
            checkpoints_dir=checkpoints_dir,
        )
        result = worker.execute_work_unit(work_unit, trial, enable_checkpointing)
        return work_unit, result
    except Exception:
        return work_unit, {"error": traceback.format_exc()}


class ComputeScheduler:
    """
    Manages a pool of worker processes using concurrent.futures. It submits
    work units and yields results as they are completed, using the
    `as_completed` pattern for efficiency. This is the 'Compute Scheduler'
    described in the architecture.
    """

    def __init__(
        self,
        datastore: DataStore,
        dataset_name: str,
        max_workers: int = 2,
        enable_checkpointing: bool = False,
        checkpoints_dir: str = "./checkpoints",
    ):
        self.datastore = datastore
        self.dataset_name = dataset_name
        self.max_workers = max_workers
        self.enable_checkpointing = enable_checkpointing
        self.checkpoints_dir = checkpoints_dir

        self.executor = None
        self._is_running = False
        self._lock = threading.Lock()

        # Futures management
        self.active_futures: Dict[concurrent.futures.Future, WorkUnit] = {}
        self.trial_to_futures: Dict[str, List[concurrent.futures.Future]] = {}

    def start(self):
        """Initializes the process pool executor."""
        if not self._is_running:
            ctx = multiprocessing.get_context("spawn")
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=ctx
            )
            self._is_running = True

    def stop(self):
        """Stops the scheduler and shuts down the executor."""
        if self._is_running:
            self._is_running = False
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def cancel_work_for_trial(self, trial_id: str):
        """Cancels all active and pending futures for a specific trial."""
        with self._lock:
            if trial_id in self.trial_to_futures:
                futures_to_cancel = self.trial_to_futures.pop(trial_id)
                cancelled_count = 0
                for future in futures_to_cancel:
                    if future.cancel():
                        cancelled_count += 1
                    # Also remove from the primary map
                    if future in self.active_futures:
                        del self.active_futures[future]
                logger.info(
                    f"Cancelled {cancelled_count}/{len(futures_to_cancel)} futures for trial {trial_id}."
                )

    def run(self, work_units: List[WorkUnit]) -> Iterator[Tuple[WorkUnit, dict]]:
        """
        Submits work units to the executor and yields results as they complete.
        This is a generator function.
        """
        if not self._is_running or not self.executor:
            raise RuntimeError("Scheduler must be started before running work.")

        with self._lock:
            for work_unit in work_units:
                trial = self.datastore.get_trial(work_unit.trial_id)
                if not trial or trial.status != TrialStatus.ACTIVE:
                    continue

                future = self.executor.submit(
                    execute_work_unit_in_process,
                    work_unit,
                    trial,
                    trial.algorithm_name,
                    self.dataset_name,
                    self.enable_checkpointing,
                    self.checkpoints_dir,
                )
                self.active_futures[future] = work_unit
                self.trial_to_futures.setdefault(trial.id, []).append(future)

        # Use a copy of the keys for safe iteration, as the dictionary can be
        # modified by cancel_work_for_trial
        active_futures_copy = list(self.active_futures.keys())
        for future in concurrent.futures.as_completed(active_futures_copy):
            if not self._is_running:
                break

            work_unit = self.active_futures.get(future)
            if not work_unit:
                # This can happen if the future was cancelled and removed
                logger.warning(
                    f"Future {future} completed but was not in the active map, likely cancelled."
                )
                continue

            try:
                _, result = future.result()
                yield work_unit, result
            except concurrent.futures.CancelledError:
                logger.warning(
                    f"Work unit for trial {work_unit.trial_id} was cancelled."
                )
                continue
            except Exception:
                yield work_unit, {"error": traceback.format_exc()}
            finally:
                # Clean up finished future from our tracking maps
                with self._lock:
                    if future in self.active_futures:
                        del self.active_futures[future]
                    if work_unit.trial_id in self.trial_to_futures:
                        if future in self.trial_to_futures[work_unit.trial_id]:
                            self.trial_to_futures[work_unit.trial_id].remove(future)
                        if not self.trial_to_futures[work_unit.trial_id]:
                            del self.trial_to_futures[work_unit.trial_id]
