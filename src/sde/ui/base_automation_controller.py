from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer

if TYPE_CHECKING:
    from .main_window import MainWindow


class BaseAutomationController(ABC):
    """Abstract base class for controllers that automate UI interactions."""

    def __init__(self, main_window: "MainWindow", update_interval: int = 5000):
        self.main_window = main_window
        self.view_model = main_window.view_model
        self._timer = QTimer(main_window)
        self._timer.timeout.connect(self.on_tick)
        self._update_interval = update_interval

    @abstractmethod
    def start(self):
        """Starts the automation sequence."""
        self._timer.start(self._update_interval)

    def stop(self):
        """Stops the automation sequence."""
        self._timer.stop()

    @abstractmethod
    def on_tick(self):
        """The main logic loop called on each timer tick."""
        pass