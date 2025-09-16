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
    prioritized: bool = False
    # This color will be assigned by the ViewModel to ensure uniqueness for the plot
    plot_color: QColor = field(default_factory=lambda: DEFAULT_COLOR)

    @property
    def pen(self) -> pg.QtGui.QPen:
        """Determines the pen style for this trial's plot curve based on its status."""
        color = self.plot_color
        style = pg.QtCore.Qt.PenStyle.SolidLine
        width = 2

        # The 'best' trial gets a unique, standout style
        if self.is_best:
            pen = pg.mkPen(color="#FFD700", width=4)  # Gold
            pen.setCosmetic(True)  # Ensures width is consistent when zooming
            return pen

        # Other statuses modify the base plot color and style
        if self.status == "ACTIVE":
            width = 3  # Make active trials slightly thicker
        elif self.status == "PRUNED":
            style = pg.QtCore.Qt.PenStyle.DotLine
            color.setAlphaF(0.6)  # Make it semi-transparent
        elif self.status == "FAILED":
            style = pg.QtCore.Qt.PenStyle.DashLine
            color.setAlphaF(0.7)
        elif self.status == "COMPLETED":
            # Completed trials are solid but slightly less prominent than active ones
            width = 2
        elif self.status == "PENDING":
            width = 1
            style = pg.QtCore.Qt.PenStyle.DotLine

        pen = pg.mkPen(color=color, width=width, style=style)
        pen.setCosmetic(True)
        return pen

    @property
    def row_background_color(self) -> QColor:
        """Determines a subtle background color for this trial's row in the table."""
        if self.is_best:
            return QColor("#FFFACD")  # LemonChiffon (for gold)
        if self.prioritized:
            return QColor("#E6F7FF")  # Very light blue for prioritized trials

        # Colors are subtle to avoid a "rainbow" effect and keep focus on the data
        status_colors = {
            "ACTIVE": QColor("#E9FEE9"),  # A hint of green
            "COMPLETED": QColor("#F0F8FF"),  # AliceBlue
            "PRUNED": QColor("#F5F5F5"),  # A light gray (WhiteSmoke)
            "FAILED": QColor("#FFF0F0"),  # A hint of red (Snow)
            "PENDING": QColor("white"),
        }
        return status_colors.get(self.status, QColor("white"))

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
