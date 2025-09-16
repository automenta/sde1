import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from sde.core.comms import EngineCommand
from sde.core.comms import EngineEvent

from .runtime import SdeRuntimeEngine

logger = logging.getLogger(__name__)


class EngineProxy:
    """A proxy that sits between the Orchestrator and the SdeRuntimeEngine,
    translating method calls into commands and forwarding events.
    This allows the main application thread and the engine thread to be
    cleanly decoupled.
    """

    def __init__(self, event_callback: Callable[[EngineEvent, Dict[str, Any]], None]):
        """Args:
        event_callback: A callback function in the Orchestrator to which
                        the proxy will forward all events from the engine.

        """
        self.orchestrator_callback = event_callback
        self.runtime_engine = SdeRuntimeEngine(self._on_event_from_engine)
        logger.info("EngineProxy initialized.")

    def _on_event_from_engine(self, event_type: EngineEvent, payload: Dict[str, Any]):
        """Receives an event from the engine and forwards it to the orchestrator."""
        # This method is called by the SdeRuntimeEngine's thread.
        # It passes the event along to the orchestrator's callback.
        self.orchestrator_callback(event_type, payload)

    def post_command(
        self, command_type: EngineCommand, payload: Optional[Dict[str, Any]] = None
    ):
        """A generic method to post a command to the engine."""
        if payload is None:
            payload = {}
        self.runtime_engine.post_command(command_type, payload)

    def shutdown(self):
        """Shuts down the runtime engine gracefully."""
        # The STOP command will handle the thread shutdown.
        self.post_command(EngineCommand.STOP_RUN)
        logger.info("EngineProxy shutdown command sent.")
