"""
Manages the lifecycle of the SdeRuntimeEngine.
"""
from __future__ import annotations
import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional

from sde.core.domain import Challenge, ExecutionSettings, PatienceBudget, Trial, TrialStatus
from sde.engine.datastore import DataStore
from sde.engine.factory import SchedulerFactory
from sde.engine.insight import InsightEngine
from sde.engine.compute_scheduler import ComputeScheduler

logger = logging.getLogger(__name__)


class RuntimeStatus(Enum):
    """Define the lifecycle status of the runtime engine."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class RuntimeLifecycleManager:
    """Manages the lifecycle of the SdeRuntimeEngine."""

    def __init__(self, runtime_engine: "SdeRuntimeEngine", emit_event: Callable):
        self.runtime_engine = runtime_engine
        self.emit_event = emit_event
        self._status = RuntimeStatus.IDLE
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def status(self):
        return self._status

    def start_run(self, payload: Dict[str, Any]):
        """Handle the START_RUN command."""
        if self._status != RuntimeStatus.IDLE:
            logger.warning(f"START_RUN received, but engine is in state {self._status.name}.")
            return

        self.runtime_engine.initialize_components(payload)

        start_paused = payload.get("start_paused", False)

        assert self.runtime_engine.datastore is not None
        assert self.runtime_engine.adaptive_scheduler is not None
        assert self.runtime_engine.compute_scheduler is not None
        assert self.runtime_engine.work_manager is not None

        all_trials = self.runtime_engine.datastore.get_all_trials()
        is_resumed_run = any(t.status != TrialStatus.PENDING for t in all_trials.values())

        if is_resumed_run:
            logger.info("Resuming experiment. Rehydrating work queue...")
            work_units = self.runtime_engine.adaptive_scheduler.rehydrate_work_units(all_trials)
        else:
            logger.info("Starting fresh experiment. Generating initial work units...")
            pending_trials = [t for t in all_trials.values() if t.status == TrialStatus.PENDING]
            exp_def = payload.get("experiment_definition", {})
            work_units, scheduler_state = self.runtime_engine.adaptive_scheduler.get_initial_work_units(
                pending_trials, exp_def.get("scheduler_state", {})
            )
            exp_def["scheduler_state"] = scheduler_state

        for work_unit in work_units:
            self.runtime_engine.work_manager.put_work_in_queue(work_unit)

        self.runtime_engine.compute_scheduler.start()
        self._status = RuntimeStatus.PAUSED if start_paused else RuntimeStatus.RUNNING
        if not start_paused:
            self._pause_event.set()

        self._thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._thread.start()
        logger.info("SdeRuntimeEngine started successfully.")

        initial_trials_dict = [t.to_dict() for t in self.runtime_engine.datastore.get_all_trials().values()]
        from sde.core.events import EngineEvent
        self.emit_event(EngineEvent.RUN_STARTED, {"trials": initial_trials_dict})

    def stop_run(self):
        """Handle the STOP_RUN command."""
        if self._status in [RuntimeStatus.STOPPING, RuntimeStatus.STOPPED]:
            return

        self._status = RuntimeStatus.STOPPING
        self._pause_event.set()
        if self.runtime_engine.compute_scheduler:
            self.runtime_engine.compute_scheduler.stop()

        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None
        self._status = RuntimeStatus.STOPPED
        logger.info("SdeRuntimeEngine has been cleanly shut down.")
        from sde.core.events import EngineEvent
        self.emit_event(EngineEvent.RUN_STOPPED, {})

    def pause_run(self):
        """Handle the PAUSE_RUN command."""
        if self._status == RuntimeStatus.RUNNING:
            self._status = RuntimeStatus.PAUSED
            self._pause_event.clear()
            logger.info("Runtime engine execution paused.")
            from sde.core.events import EngineEvent
            self.emit_event(EngineEvent.RUN_PAUSED, {})

    def resume_run(self):
        """Handle the RESUME_RUN command."""
        if self._status == RuntimeStatus.PAUSED:
            self._status = RuntimeStatus.RUNNING
            self._pause_event.set()
            logger.info("Runtime engine execution resumed.")
            from sde.core.events import EngineEvent
            self.emit_event(EngineEvent.RUN_RESUMED, {})

    def _execution_loop(self):
        """Run the main loop that drives the experiment."""
        logger.info("Runtime engine execution loop started.")
        while self._status not in [RuntimeStatus.STOPPING, RuntimeStatus.STOPPED]:
            self._pause_event.wait()
            if self._status in [RuntimeStatus.STOPPING, RuntimeStatus.STOPPED]:
                break

            self.runtime_engine.command_processor.process_commands()

            if self._should_stop_or_wait():
                break

            self.runtime_engine.work_manager.run_work_batch()

        self._status = RuntimeStatus.STOPPED
        logger.info("Runtime engine execution loop finished.")

    def _should_stop_or_wait(self) -> bool:
        """Check if the execution loop should stop or wait for more work."""
        if not self.runtime_engine.work_manager.is_work_queue_empty():
            return False

        if self._is_run_complete():
            logger.info("All trials are finished. Shutting down.")
            self.stop_run()
            return True

        threading.Event().wait(0.5)
        return False

    def _is_run_complete(self) -> bool:
        """Check if all trials in the datastore are in a terminal state."""
        if not self.runtime_engine.datastore or not self.runtime_engine.datastore.get_all_trials():
            return False
        return all(
            t.status in (TrialStatus.COMPLETED, TrialStatus.PRUNED, TrialStatus.FAILED)
            for t in self.runtime_engine.datastore.get_all_trials().values()
        )
