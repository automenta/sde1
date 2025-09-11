from typing import Optional
from typing import List

from PyQt6.QtCore import pyqtProperty, QPropertyAnimation, pyqtSignal, QEasingCurve
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QListWidget, QListWidgetItem, QWidget

from ..models import UIInsight
from ..view_model import ExperimentViewModel


class InsightListItem(QListWidgetItem):
    """A custom QListWidgetItem that stores the full UIInsight object."""

    def __init__(
        self, ui_insight: UIInsight, parent: Optional[QListWidget] = None
    ):
        super().__init__(parent)
        self.insight = ui_insight
        self.setIcon(ui_insight.icon)
        self.setText(f"[{ui_insight.timestamp}] {ui_insight.message}")
        self.setToolTip(ui_insight.message)


class InsightsWidget(QWidget):
    """A widget for displaying insights."""

    insight_selected = pyqtSignal(object)  # Emits InsightListItem

    def __init__(self, parent=None):
        super().__init__(parent)
        self.displayed_insight_count = 0
        self.insight_animation: Optional[QPropertyAnimation] = None
        self.original_insights_stylesheet: Optional[str] = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initializes the main UI layout and sub-components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.insights_group = QGroupBox("Insights")
        insights_layout = QVBoxLayout(self.insights_group)
        self.insights_list = QListWidget()
        self.insights_list.setWordWrap(True)
        insights_layout.addWidget(self.insights_list)
        insights_layout.setContentsMargins(0, 5, 0, 0)
        self.insights_group.setLayout(insights_layout)

        self.original_insights_stylesheet = self.insights_group.styleSheet()
        main_layout.addWidget(self.insights_group)

    def _connect_signals(self):
        """Connects internal widget signals."""
        self.insights_list.itemClicked.connect(self.insight_selected)

    def update_insights_list(self, view_model: ExperimentViewModel):
        """Updates the insights list from the ViewModel efficiently."""
        num_new_insights = len(view_model.insights) - self.displayed_insight_count
        if num_new_insights <= 0:
            return

        new_insights = view_model.insights[-num_new_insights:]
        for ui_insight in new_insights:
            item = InsightListItem(ui_insight, self.insights_list)
            self.insights_list.addItem(item)

        self.displayed_insight_count = len(view_model.insights)
        self.insights_list.scrollToBottom()

        if num_new_insights > 0:
            self._trigger_insight_animation()

    def _trigger_insight_animation(self):
        """Animates the border and title of the 'Insights' group box to signal a new insight."""
        if self.insight_animation and self.insight_animation.state() == QPropertyAnimation.State.Running:
            self.insight_animation.stop()

        self.insight_animation = QPropertyAnimation(self, b"insightBorderColor")
        self.insight_animation.setDuration(2000)
        self.insight_animation.setStartValue(QColor("#FFD700"))
        self.insight_animation.setEndValue(QColor(0, 0, 0, 0))
        self.insight_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.insight_animation.finished.connect(self._reset_insight_style)
        self.insight_animation.start()

    def _reset_insight_style(self):
        """Resets the insights group box to its original, default stylesheet."""
        self.insights_group.setStyleSheet(self.original_insights_stylesheet)

    def _set_insight_border_color(self, color: QColor):
        """Sets a prominent border and title background color for the insights group box."""
        text_color = "black" if color.lightnessF() > 0.5 else "white"
        self.insights_group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {color.name(QColor.NameFormat.HexArgb)};
                margin-top: 1em;
                border-radius: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: {color.name(QColor.NameFormat.HexArgb)};
                color: {text_color};
                border-radius: 4px;
            }}
        """)

    insightBorderColor = pyqtProperty(QColor, fset=_set_insight_border_color)

    def clear(self):
        """Clears the insights list."""
        self.insights_list.clear()
        self.insights_group.setStyleSheet("")  # Reset stylesheet
        self.displayed_insight_count = 0

    def get_selected_item(self) -> Optional[InsightListItem]:
        """Returns the currently selected insight item."""
        selected_items = self.insights_list.selectedItems()
        return selected_items[0] if selected_items else None

    def clear_selection(self):
        """Clears the selection in the insights list."""
        self.insights_list.clearSelection()
