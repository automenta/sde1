"""Module for the SDE runtime engine, the computational core."""
from __future__ import annotations
import logging
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple

from sde.core.domain import Challenge, ExecutionSettings, PatienceBudget, Trial
from sde.exploration.schedulers import AdaptiveScheduler
from sde.core.events import EngineCommand, EngineEvent
from sde.engine.datastore import DataStore
from sde.engine.factory import SchedulerFactory
from sde.engine.insight import InsightEngine
from sde.engine.compute_scheduler import ComputeScheduler
from .runtime_command_processor import RuntimeCommandProcessor
from .runtime_lifecycle_manager import RuntimeLifecycleManager
from .runtime_work_manager import RuntimeWorkManager

logger = logging.getLogger(__name__)


class SdeRuntimeEngine:
    """A facade that coordinates the components of the runtime engine."""

    def __init__(self, event_callback: Callable[[EngineEvent, Dict[str, Any]], None]):
        self.event_callback = event_callback
        self.command_queue: queue.Queue[Tuple[EngineCommand, Dict[str, Any]]] = queue.Queue()

        # Core components are initialized on START_RUN
        self.datastore: Optional[DataStore] = None
        self.adaptive_scheduler: Optional[AdaptiveScheduler] = None
        self.insight_engine: Optional[InsightEngine] = None
        self.compute_scheduler: Optional[ComputeScheduler] = None

        # Delegated responsibilities
        self.lifecycle_manager = RuntimeLifecycleManager(self, self._emit_event)
        self.command_processor = RuntimeCommandProcessor(self, self.command_queue, self._emit_event)
        self.work_manager: Optional[RuntimeWorkManager] = None

    def _emit_event(self, event_type: EngineEvent, payload: Dict[str, Any]):
        self.event_callback(event_type, payload)

    def post_command(self, command: EngineCommand, payload: Dict[str, Any]) -> None:
        self.command_queue.put((command, payload))

    def initialize_components(self, payload: Dict[str, Any]):
        """Initializes all the core components of the engine."""
        exp_def = payload.get("experiment_definition", {})
        exec_settings_dict = payload.get("execution_settings")

        trials = [Trial.from_dict(t) for t in exp_def.get("trials", {}).values()]
        execution_settings = ExecutionSettings.from_dict(exec_settings_dict) if exec_settings_dict else ExecutionSettings()
        challenge = Challenge.from_dict(exp_def.get("challenge"))
        from sde.registry import registry
        challenge_def = registry.get_challenge(challenge.name)
        patience_budget_dict = exp_def.get("patience_budget")
        patience_budget = PatienceBudget.from_dict(patience_budget_dict) if patience_budget_dict is not None else PatienceBudget()

        self.datastore = DataStore(trials)
        self.adaptive_scheduler = SchedulerFactory.create_scheduler(
            policy_name=exp_def.get("adaptive_policy"),
            challenge_name=challenge_def.name,
            patience_budget=patience_budget.to_dict(),
        )
        self.insight_engine = InsightEngine(
            self.datastore.get_all_trials(),
            primary_metric=self.adaptive_scheduler.metric,
            higher_is_better=self.adaptive_scheduler.increasing,
        )
        self.compute_scheduler = ComputeScheduler(
            datastore=self.datastore,
            dataset_def=challenge_def,
            max_workers=execution_settings.num_workers,
            enable_checkpointing=execution_settings.enable_checkpointing,
            work_unit_timeout=execution_settings.work_unit_timeout_seconds,
        )
        self.work_manager = RuntimeWorkManager(
            self.datastore, self.compute_scheduler, self.adaptive_scheduler, self.insight_engine, self._emit_event
        )
        logger.info(f"Runtime engine components initialized with {exp_def.get('adaptive_policy')} scheduler.")

    def start_run(self, payload: Dict[str, Any]):
        self.lifecycle_manager.start_run(payload)

    def stop_run(self):
        self.lifecycle_manager.stop_run()

    def pause_run(self):
        self.lifecycle_manager.pause_run()

    def resume_run(self):
        self.lifecycle_manager.resume_run()

    def cancel_work_for_trial(self, trial_id: str):
        if self.work_manager:
            self.work_manager.cancel_work_for_trial(trial_id)
