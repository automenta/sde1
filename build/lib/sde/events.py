import logging
import traceback
from typing import Callable
from typing import List

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
