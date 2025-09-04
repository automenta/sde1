import threading
import logging
from typing import List, Callable, Iterator, Tuple

from sde.core.types import Trial, WorkUnit
from sde.engine.datastore import DataStore
from sde.engine.scheduler import Scheduler
from sde.engine.insight import InsightEngine
from sde.exploration.schedulers import AdaptiveScheduler

logger = logging.getLogger(__name__)

class SdeRuntimeEngine:
    """
    Wraps the core computational components (Scheduler, DataStore, etc.)
    and exposes a simple API to the Orchestrator. This is the "Engine Room".
    """

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

    def start(self):
        """Starts the underlying scheduler."""
        if not self._is_running:
            self._is_running = True
            self.scheduler.start()

    def stop(self):
        """Stops the underlying scheduler."""
        if self._is_running:
            self._is_running = False
            self.scheduler.stop()

    def submit_work(self, work_units: List[WorkUnit]) -> Iterator[Tuple[WorkUnit, dict]]:
        """Submits a list of work units to the scheduler and yields results."""
        if not self._is_running:
            raise RuntimeError("Runtime Engine is not running.")
        return self.scheduler.run(work_units)
