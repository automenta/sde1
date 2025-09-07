from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pyqtgraph as pg
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QIcon

# A unique, consistent color can be assigned by the ViewModel.
# This default is just a fallback.
DEFAULT_COLOR = QColor("black")


@dataclass
class UITrial:
    """UI-specific data model for a Trial."""

    id: str
    algorithm_name: str
    status: str
    current_epoch: int
    hyperparameters: Dict[str, Any]
    results: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    est_time_per_epoch: Optional[float] = None
    is_best: bool = False
    # This color will be assigned by the ViewModel to ensure uniqueness for the plot
    plot_color: QColor = field(default_factory=lambda: DEFAULT_COLOR)

    @property
    def pen(self) -> pg.QtGui.QPen:
        """Determines the pen style for this trial's plot curve."""
        # Start with the base color assigned by the ViewModel
        color = self.plot_color
        style = pg.QtCore.Qt.PenStyle.SolidLine
        width = 2

        if self.is_best:
            color = pg.mkColor("#FFD700")  # Gold
            width = 4
        elif self.status == "FAILED":
            color = pg.mkColor("#DC143C")  # Crimson
            style = pg.QtCore.Qt.PenStyle.DotLine
        elif self.status == "PRUNED":
            color = pg.mkColor("#808080")  # Gray
            style = pg.QtCore.Qt.PenStyle.DotLine
        elif self.status == "COMPLETED":
            color = pg.mkColor("#0000FF")  # Blue

        pen = pg.mkPen(color=color, width=width)
        pen.setStyle(style)
        return pen

    @property
    def row_background_color(self) -> QColor:
        """Determines the background color for this trial's row in the table."""
        if self.is_best:
            return QColor("#FFFACD")  # LemonChiffon
        if self.status == "FAILED":
            return QColor("#F08080")  # LightCoral
        if self.status == "PRUNED":
            return QColor("#D3D3D3")  # LightGray
        if self.status == "COMPLETED":
            return QColor("#ADD8E6")  # LightBlue
        return QColor("white")

    def get_latest_metric(self, metric_name: str) -> str:
        """Formats the latest metric value for display."""
        metric_list = self.results.get(metric_name, [])
        return f"{metric_list[-1][1]:.4f}" if metric_list else "N/A"

    @property
    def display_epoch(self) -> str:
        """Returns the current epoch as a string for display."""
        return str(self.current_epoch)

    @property
    def display_est_time(self) -> str:
        """Formats the estimated time per epoch for display."""
        return (
            f"{self.est_time_per_epoch:.2f}s"
            if self.est_time_per_epoch is not None
            else "N/A"
        )


@dataclass
class UIAlgorithm:
    """UI-specific data model for an Algorithm."""

    id: str
    name: str


@dataclass
class UIInsight:
    """UI-specific data model for an Insight."""

    message: str
    type: str
    trial_ids: List[str]
    timestamp: str
    icon: QIcon
    # The full insight object from the backend, for reference if needed
    raw_insight: Any
