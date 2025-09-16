from typing import Dict
from typing import Optional
from typing import Set

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from ..models import UITrial
from ..view_model import ExperimentViewModel
from .base import VisualizationPlugin


class ClickableLabelItem(pg.LabelItem):
    """A LabelItem that emits a signal when clicked."""

    clicked = pyqtSignal(object, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curve = None
        self.label = None

    def mouseClickEvent(self, ev):
        self.clicked.emit(self.curve, self.label)


class CustomLegendItem(pg.LegendItem):
    """A LegendItem that uses ClickableLabelItems and emits a signal when an item is clicked."""

    itemClicked = pyqtSignal(object, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def addItem(self, item, name):
        """Overrides the default addItem to use a ClickableLabelItem."""
        label = ClickableLabelItem(
            text=name,
            color=self.opts["labelTextColor"],
            size=self.opts["labelTextSize"],
        )
        label.curve = item
        label.label = label
        label.clicked.connect(self.itemClicked.emit)
        sample = pg.graphicsItems.LegendItem.ItemSample(item)
        self.items.append((sample, label))
        self._updateLayout()

    def _updateLayout(self):
        """A simplified layout update. Assumes single column."""
        for i in range(self.layout.count()):
            self.layout.removeAt(0)
        for sample, label in self.items:
            row = self.layout.rowCount()
            self.layout.addItem(sample, row, 0)
            self.layout.addItem(label, row, 1)
        self.update()


class PerformancePlotWidget(QWidget):
    """A widget for displaying trial performance plots."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.plot_curve_map: Dict[str, pg.PlotDataItem] = {}
        self.legend: Optional[CustomLegendItem] = None
        self.view_model: Optional[ExperimentViewModel] = None
        self.available_metrics: Set[str] = set()
        self.plot_curve_visibility: Dict[str, bool] = {}
        self.highlighted_trial_ids: Set[str] = set()

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.setup_plot()

        metric_selection_layout = QHBoxLayout()
        metric_selection_layout.addStretch()
        metric_label = QLabel("<b>Plotting Metric:</b>")
        metric_selection_layout.addWidget(metric_label)
        self.metric_combo = QComboBox()
        self.metric_combo.setMinimumWidth(150)
        metric_selection_layout.addWidget(self.metric_combo)

        layout.addLayout(metric_selection_layout)
        layout.addWidget(self.plot_widget)

    def _connect_signals(self):
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        if self.legend:
            self.legend.itemClicked.connect(self._on_legend_item_clicked)

    def set_view_model(self, view_model: ExperimentViewModel):
        self.view_model = view_model

    def setup_plot(self):
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("Real-Time Trial Performance", color="k", size="16pt")
        self.plot_widget.setLabel(
            "left", "Accuracy", color="k", **{"font-size": "12pt"}
        )
        self.plot_widget.setLabel("bottom", "Epoch", color="k", **{"font-size": "12pt"})
        self.plot_widget.showGrid(x=True, y=True)
        self.legend = CustomLegendItem()
        self.legend.setParentItem(self.plot_widget.getPlotItem())
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))
        self.plot_curve_visibility = {}

    def update_plots(self, trials: Dict[str, UITrial]):
        if not self.view_model:
            return

        metric_name = (
            self.metric_combo.currentText() or self.view_model.performance_metric_name
        )
        self._update_available_metrics(trials)

        current_trial_ids = set(trials.keys())
        cached_plot_ids = set(self.plot_curve_map.keys())

        for trial_id in cached_plot_ids - current_trial_ids:
            self.plot_widget.removeItem(self.plot_curve_map.pop(trial_id))

        for trial_id, ui_trial in trials.items():
            if trial_id not in self.plot_curve_map:
                name = f"{ui_trial.algorithm_name} ({trial_id[:6]})"
                pen = ui_trial.pen
                self.plot_curve_map[trial_id] = self.plot_widget.plot(
                    [],
                    [],
                    name=name,
                    pen=pen,
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=pen.color(),
                )

            metric_list = ui_trial.results.get(metric_name, [])
            if metric_list:
                try:
                    epochs, metrics = zip(*metric_list)
                    self.plot_curve_map[trial_id].setData(epochs, metrics)
                except ValueError:
                    self.plot_curve_map[trial_id].clear()

        self._apply_plot_curve_styles()

    def update_plot_highlight(self, highlight_ids: set):
        self.highlighted_trial_ids = highlight_ids
        self._apply_plot_curve_styles()

    def _on_metric_changed(self):
        if not self.view_model:
            return

        metric_name = self.metric_combo.currentText()
        if not metric_name:
            return

        self.plot_widget.setLabel("left", metric_name.replace("_", " ").title())
        self.plot_widget.setTitle(
            f"Real-Time Trial Performance: {metric_name.replace('_', ' ').title()}",
            color="k",
            size="16pt",
        )

        for trial_id, curve in self.plot_curve_map.items():
            ui_trial = self.view_model.trials.get(trial_id)
            if ui_trial:
                metric_list = ui_trial.results.get(metric_name, [])
                if metric_list:
                    try:
                        epochs, metrics = zip(*metric_list)
                        curve.setData(epochs, metrics)
                    except ValueError:
                        curve.clear()
                else:
                    curve.clear()

    def _update_available_metrics(self, trials: Dict[str, UITrial]):
        new_metrics: Set[str] = set()
        for trial in trials.values():
            new_metrics.update(trial.results.keys())

        if new_metrics != self.available_metrics:
            self.available_metrics = new_metrics
            current_selection = self.metric_combo.currentText()
            self.metric_combo.blockSignals(True)
            self.metric_combo.clear()
            sorted_metrics = sorted(list(self.available_metrics))
            if sorted_metrics:
                self.metric_combo.addItems(sorted_metrics)
                if current_selection in sorted_metrics:
                    self.metric_combo.setCurrentText(current_selection)
                elif (
                    self.view_model
                    and self.view_model.performance_metric_name in sorted_metrics
                ):
                    self.metric_combo.setCurrentText(
                        self.view_model.performance_metric_name
                    )
            self.metric_combo.blockSignals(False)

    def _on_legend_item_clicked(self, curve_item, label_item):
        clicked_trial_id = None
        for trial_id, curve in self.plot_curve_map.items():
            if curve is curve_item:
                clicked_trial_id = trial_id
                break

        if clicked_trial_id:
            self.plot_curve_visibility[clicked_trial_id] = (
                not self.plot_curve_visibility.get(clicked_trial_id, True)
            )
            self._apply_plot_curve_styles()

    def _apply_plot_curve_styles(self):
        if not self.view_model:
            return

        for trial_id, curve in self.plot_curve_map.items():
            ui_trial = self.view_model.trials.get(trial_id)
            if not ui_trial:
                continue

            pen = ui_trial.pen
            color = pen.color()
            is_visible = self.plot_curve_visibility.get(trial_id, True)
            is_highlighted = trial_id in self.highlighted_trial_ids

            if is_highlighted:
                color.setAlpha(255)
                curve.setPen(pg.mkPen(color=color, width=4))
                curve.setZValue(100)
            elif is_visible:
                color.setAlpha(200)
                curve.setPen(pg.mkPen(color=color, width=2))
                curve.setZValue(0)
            else:
                color.setAlpha(15)
                curve.setPen(pg.mkPen(color=color, width=1, style=Qt.PenStyle.DotLine))
                curve.setZValue(-100)

            for _, label in self.legend.items:
                if label.text == curve.name():
                    label.setText(label.text, color="k" if is_visible else "gray")
                    break

    def clear(self):
        self.plot_widget.clear()
        self.plot_curve_map.clear()
        self.available_metrics.clear()
        self.metric_combo.clear()
        self.setup_plot()


class PerformancePlotPlugin(VisualizationPlugin):
    """Plugin for the performance plot."""

    @property
    def name(self) -> str:
        return "Performance Plot"

    def create_widget(
        self, parent: Optional[QWidget], view_model: ExperimentViewModel
    ) -> QWidget:
        # This is not ideal, we are creating a new widget every time.
        # A better approach would be to cache the widget.
        # But for now, this is fine.
        widget = PerformancePlotWidget(parent)
        widget.set_view_model(view_model)
        return widget
