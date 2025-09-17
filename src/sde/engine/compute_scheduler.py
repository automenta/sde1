import concurrent.futures
import logging
import multiprocessing
import threading
import traceback
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple

from sde.core.definitions import DatasetDefinition
from sde.core.definitions import ModelDefinition
from sde.core.domain import Trial
from sde.core.domain import TrialStatus
from sde.core.domain import WorkUnit
from sde.engine.datastore import DataStore
from sde.registry import registry

logger = logging.getLogger(__name__)


def execute_work_unit_in_process(
    work_unit: WorkUnit,
    trial: Trial,
    model_def: ModelDefinition,
    dataset_def: DatasetDefinition,
    enable_checkpointing: bool,
) -> Tuple[WorkUnit, dict]:
    """A wrapper function that initializes a Worker in a new process, executes
    the given WorkUnit, and returns the result along with the original WorkUnit.
    It's a top-level function to be pickleable.
    """
    try:
        from .worker import Worker  # Import inside for pickling

        worker = Worker(model_def=model_def, dataset_def=dataset_def)
        result = worker.execute_work_unit(work_unit, trial, enable_checkpointing)
        return work_unit, result
    except Exception:
        return work_unit, {"error": traceback.format_exc()}


class ComputeScheduler:
    """Manages a pool of worker processes using concurrent.futures. It submits
    work units and yields results as they are completed, using the
    `as_completed` pattern for efficiency. This is the 'Compute Scheduler'
    described in the architecture.
    """

    def __init__(
        self,
        datastore: DataStore,
        dataset_def: DatasetDefinition,
        max_workers: int = 2,
        enable_checkpointing: bool = False,
        work_unit_timeout: int = 300,
    ):
        self.datastore = datastore
        self.dataset_def = dataset_def
        self.max_workers = max_workers
        self.enable_checkpointing = enable_checkpointing
        self.work_unit_timeout = work_unit_timeout

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
        if not self._is_running:
            return

        self._is_running = False
        if self.executor:
            # Instruct the executor to shutdown. `cancel_futures=True` is a new
            # robust addition to cancel queued work that hasn't started.
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor = None
        logger.info("ComputeScheduler has been stopped and executor shut down.")

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
                    f"Cancelled {cancelled_count}/{len(futures_to_cancel)} futures for "
                    f"trial {trial_id}."
                )

    def run(
        self, work_units: List[WorkUnit]
    ) -> Iterator[Tuple[WorkUnit, Dict[str, Any]]]:
        """Submit work units to the executor and yield results as they complete."""
        if not self._is_running or not self.executor:
            raise RuntimeError("Scheduler must be started before running work.")

        submitted_futures = self._submit_work_units(work_units)

        for future in concurrent.futures.as_completed(submitted_futures):
            if not self._is_running:
                break

            work_unit = self.active_futures.get(future)
            if not work_unit:
                logger.warning(
                    f"Future {future} completed but was not in the active map, "
                    f"likely cancelled."
                )
                continue

            try:
                _, result = future.result(timeout=self.work_unit_timeout)
                yield work_unit, result
            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"Work unit for trial {work_unit.trial_id} timed out after "
                    f"{self.work_unit_timeout} seconds."
                )
                yield (
                    work_unit,
                    {
                        "error": (
                            f"Work unit timed out after {self.work_unit_timeout} "
                            "seconds."
                        )
                    },
                )
            except concurrent.futures.CancelledError:
                logger.warning(
                    f"Work unit for trial {work_unit.trial_id} was cancelled."
                )
            except Exception:
                yield work_unit, {"error": traceback.format_exc()}
            finally:
                self._cleanup_future(future)

    def _submit_work_units(self, work_units: List[WorkUnit]) -> List:
        """Submit a list of work units to the executor and return the futures."""
        futures = []
        with self._lock:
            for work_unit in work_units:
                trial = self.datastore.get_trial(work_unit.trial_id)
                if not trial or trial.status != TrialStatus.ACTIVE:
                    continue

                model_def = registry.get_model(trial.algorithm_name)
                future = self.executor.submit(
                    execute_work_unit_in_process,
                    work_unit,
                    trial,
                    model_def,
                    self.dataset_def,
                    self.enable_checkpointing,
                )
                self.active_futures[future] = work_unit
                self.trial_to_futures.setdefault(trial.id, []).append(future)
                futures.append(future)
        return futures

    def _cleanup_future(self, future: concurrent.futures.Future):
        """Remove a completed or cancelled future from all tracking maps."""
        with self._lock:
            work_unit = self.active_futures.pop(future, None)
            if work_unit and work_unit.trial_id in self.trial_to_futures:
                self.trial_to_futures[work_unit.trial_id].remove(future)
                if not self.trial_to_futures[work_unit.trial_id]:
                    del self.trial_to_futures[work_unit.trial_id]
