from abc import ABC, abstractmethod
from typing import Optional

from PyQt6.QtWidgets import QWidget

from ..view_model import ExperimentViewModel


class VisualizationPlugin(ABC):
    """Abstract base class for a visualization plugin."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The display name of the visualization plugin."""
        pass

    @abstractmethod
    def create_widget(
        self, parent: Optional[QWidget], view_model: ExperimentViewModel
    ) -> QWidget:
        """Creates and returns the main widget for this visualization."""
        pass
