"""
Handles events from the SDE Runtime Engine and mutates the experiment state.
"""
import logging
from typing import Any, Dict

from sde.core.domain import Experiment, ExperimentStatus, Trial
from sde.core.events import EngineEvent
from .insight import Insight

logger = logging.getLogger(__name__)


class EventHandler:
    """Processes events from the engine and updates the experiment state."""

    def __init__(self, experiment: Experiment, log_emitter, operation_finished_emitter):
        self.experiment = experiment
        self.log_emitter = log_emitter
        self.operation_finished_emitter = operation_finished_emitter
        self._event_handlers = self._register_event_handlers()

    def _register_event_handlers(self):
        return {
            EngineEvent.TRIAL_UPDATED: self._on_trial_updated,
            EngineEvent.INSIGHTS_GENERATED: self._on_insights_generated,
            EngineEvent.RUN_STARTED: self._on_run_started,
            EngineEvent.RUN_PAUSED: self._on_run_paused,
            EngineEvent.RUN_RESUMED: self._on_run_resumed,
            EngineEvent.RUN_STOPPED: self._on_run_stopped,
            EngineEvent.ALGORITHM_REMOVED: self._on_algorithm_removed,
            EngineEvent.OPERATION_FINISHED: self._on_operation_finished,
            EngineEvent.LOG_MESSAGE: self._on_log_message,
            EngineEvent.EXPERIMENT_LOADED: self._on_experiment_loaded,
        }

    def handle_event(self, event_type: EngineEvent, payload: Dict[str, Any]):
        """Dispatch engine events to the appropriate handler."""
        logger.info(f"EventHandler received event: {event_type.name}")
        handler = self._event_handlers.get(event_type)
        if handler:
            handler(payload)
        else:
            logger.warning(f"No handler for event type: {event_type}")

    def _on_trial_updated(self, payload: Dict[str, Any]):
        trial_data = payload["trial"]
        trial = Trial.from_dict(trial_data)
        self.experiment.trials[trial.id] = trial

    def _on_insights_generated(self, payload: Dict[str, Any]):
        self.experiment.insights.extend(
            [Insight.from_dict(d) for d in payload["insights"]]
        )

    def _on_run_started(self, payload: Dict[str, Any]):
        self.experiment.status = ExperimentStatus.RUNNING
        if "trials" in payload:
            self.experiment.trials = {
                t["id"]: Trial.from_dict(t) for t in payload["trials"]
            }
        self.log_emitter({"level": "INFO", "message": "Experiment run has started."})

    def _on_run_paused(self, payload: Dict[str, Any]):
        self.experiment.status = ExperimentStatus.PAUSED
        self.log_emitter({"level": "INFO", "message": "Experiment run has been paused."})

    def _on_run_resumed(self, payload: Dict[str, Any]):
        self.experiment.status = ExperimentStatus.RUNNING
        self.log_emitter({"level": "INFO", "message": "Experiment run has been resumed."})

    def _on_run_stopped(self, payload: Dict[str, Any]):
        self.experiment.status = ExperimentStatus.STOPPED
        self.log_emitter({"level": "INFO", "message": "Experiment run has been stopped."})

    def _on_algorithm_removed(self, payload: Dict[str, Any]):
        algo_id = payload["algorithm_id"]
        if algo_id in self.experiment.algorithms:
            del self.experiment.algorithms[algo_id]
        self.experiment.trials = {
            tid: t
            for tid, t in self.experiment.trials.items()
            if t.algorithm_name != payload["algorithm_name"]
        }
        self.log_emitter(
            {
                "level": "INFO",
                "message": f"Algorithm '{payload['algorithm_name']}' and its trials removed.",
            }
        )

    def _on_operation_finished(self, payload: Dict[str, Any]):
        self.operation_finished_emitter(payload.get("message", ""))

    def _on_log_message(self, payload: Dict[str, Any]):
        self.log_emitter(payload)

    def _on_experiment_loaded(self, payload: Dict[str, Any]):
        loaded_experiment = Experiment.from_dict(payload["experiment"])
        self.experiment = loaded_experiment  # This needs to be handled carefully
        self.log_emitter(
            {
                "level": "INFO",
                "message": f"Experiment '{loaded_experiment.id}' loaded successfully.",
            }
        )
        self.operation_finished_emitter(
            f"Experiment loaded from {payload.get('filepath', 'file')}."
        )

    def get_experiment(self) -> Experiment:
        """Returns the mutated experiment object, e.g. after loading."""
        return self.experiment
