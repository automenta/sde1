"""
Handles commands for the SdeRuntimeEngine.
"""
from __future__ import annotations
import logging
import queue
from typing import Any, Callable, Dict

from sde.core.events import EngineCommand
from sde.core.domain import Experiment, ExperimentSnapshot
from sde.engine.experiment_serializer import ExperimentSerializer

logger = logging.getLogger(__name__)


class RuntimeCommandProcessor:
    """Processes commands for the SdeRuntimeEngine."""

    def __init__(self, runtime_engine: "SdeRuntimeEngine", command_queue: queue.Queue, emit_event: Callable):
        self.runtime_engine = runtime_engine
        self.command_queue = command_queue
        self.emit_event = emit_event
        self.command_handlers = {
            EngineCommand.START_RUN: self._handle_start_run,
            EngineCommand.STOP_RUN: self._handle_stop_run,
            EngineCommand.PAUSE_RUN: self._handle_pause_run,
            EngineCommand.RESUME_RUN: self._handle_resume_run,
            EngineCommand.SAVE_EXPERIMENT: self._handle_save_experiment,
            EngineCommand.LOAD_EXPERIMENT: self._handle_load_experiment,
        }

    def process_commands(self):
        """Process all pending commands in the queue."""
        try:
            while not self.command_queue.empty():
                command, payload = self.command_queue.get_nowait()
                handler = self.command_handlers.get(command)
                if handler:
                    handler(payload)
                else:
                    logger.warning(f"Unknown command received: {command.name}")
        except queue.Empty:
            return

    def _handle_start_run(self, payload: Dict[str, Any]):
        self.runtime_engine.start_run(payload)

    def _handle_stop_run(self, payload: Dict[str, Any]):
        self.runtime_engine.stop_run()

    def _handle_pause_run(self, payload: Dict[str, Any]):
        self.runtime_engine.pause_run()

    def _handle_resume_run(self, payload: Dict[str, Any]):
        self.runtime_engine.resume_run()

    def _handle_save_experiment(self, payload: Dict[str, Any]):
        filepath = payload.get("filepath")
        experiment_data = payload.get("experiment")
        if not filepath or not experiment_data:
            return

        experiment_to_save = Experiment.from_dict(experiment_data)
        snapshot = ExperimentSnapshot(experiment=experiment_to_save)

        try:
            ExperimentSerializer.save(snapshot, filepath)
            from sde.core.events import EngineEvent
            self.emit_event(
                EngineEvent.OPERATION_FINISHED,
                {"message": f"Experiment saved to {filepath}"},
            )
        except Exception as e:
            from sde.core.events import EngineEvent
            self.emit_event(
                EngineEvent.LOG_MESSAGE,
                {"level": "ERROR", "message": f"Failed to save experiment: {e}"},
            )

    def _handle_load_experiment(self, payload: Dict[str, Any]):
        filepath = payload.get("filepath")
        if not filepath:
            return
        try:
            snapshot = ExperimentSerializer.load(filepath)
            from sde.core.events import EngineEvent
            self.emit_event(
                EngineEvent.EXPERIMENT_LOADED, {"experiment": snapshot.experiment.to_dict()}
            )
        except Exception as e:
            from sde.core.events import EngineEvent
            self.emit_event(
                EngineEvent.LOG_MESSAGE,
                {"level": "ERROR", "message": f"Failed to load experiment: {e}"},
            )
