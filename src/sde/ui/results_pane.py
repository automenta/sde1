from typing import Dict
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QSplitter
from PyQt6.QtWidgets import QTabWidget
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from .components.insights_widget import InsightsWidget
from .components.log_widget import LogWidget
from .components.trials_table_widget import TrialsTableWidget
from .models import UITrial
from .view_model import ExperimentViewModel
from .visualizations.hyperparameter_pca import HyperparameterPCAPlugin
from .visualizations.performance_plot import PerformancePlotPlugin


class ResultsPane(QWidget):
    """The right-hand pane for displaying experiment results. It acts as a
    container for the modular component widgets.
    """

    # Signals for user interactions that the parent window needs to handle
    trial_selected = pyqtSignal(str)
    trial_double_clicked = pyqtSignal(str)
    prune_trial_requested = pyqtSignal(str)
    prioritize_trial_requested = pyqtSignal(str)
    spawn_trial_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.trial_view_cache: Dict[str, UITrial] = {}
        self.view_model: Optional[ExperimentViewModel] = None
        self.visualization_plugins = []
        self.visualization_widgets = []

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initializes the main UI layout and sub-components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        self.vis_tabs = QTabWidget()
        self.load_visualization_plugins()
        for plugin in self.visualization_plugins:
            widget = plugin.create_widget(self, self.view_model)
            self.visualization_widgets.append(widget)
            self.vis_tabs.addTab(widget, plugin.name)

        bottom_pane = QWidget()
        bottom_layout = QVBoxLayout(bottom_pane)
        right_bottom_splitter = QSplitter(Qt.Orientation.Vertical)

        self.insights_widget = InsightsWidget()
        self.log_widget = LogWidget()
        right_bottom_splitter.addWidget(self.insights_widget)
        right_bottom_splitter.addWidget(self.log_widget)
        right_bottom_splitter.setSizes([100, 200])

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.trials_table_widget = TrialsTableWidget()
        bottom_splitter.addWidget(self.trials_table_widget)
        bottom_splitter.addWidget(right_bottom_splitter)
        bottom_splitter.setSizes([750, 450])
        bottom_layout.addWidget(bottom_splitter)

        splitter.addWidget(self.vis_tabs)
        splitter.addWidget(bottom_pane)
        splitter.setSizes([500, 300])

    def _connect_signals(self):
        """Connects internal widget signals to the pane's public signals."""
        self.insights_widget.insight_selected.connect(self._on_insight_selected)
        self.log_widget.refresh_requested.connect(self.refresh_requested)
        self.trials_table_widget.trial_selected.connect(self.trial_selected)
        self.trials_table_widget.trial_double_clicked.connect(self.trial_double_clicked)
        self.trials_table_widget.prune_trial_requested.connect(
            self.prune_trial_requested
        )
        self.trials_table_widget.prioritize_trial_requested.connect(
            self.prioritize_trial_requested
        )
        self.trials_table_widget.spawn_trial_requested.connect(
            self.spawn_trial_requested
        )

    def update_view(self, view_model: ExperimentViewModel):
        """The main entry point for refreshing the entire results view."""
        self.view_model = view_model
        self.trials_table_widget.set_view_model(view_model)

        for widget in self.visualization_widgets:
            if hasattr(widget, "set_view_model"):
                widget.set_view_model(view_model)
            if hasattr(widget, "update_plots"):
                widget.update_plots(view_model.trials)

        self.trials_table_widget.update_table(
            view_model.trials,
            view_model.performance_metric_name,
            self.trial_view_cache,
        )
        self.insights_widget.update_insights_list(view_model)

    def append_log_message(self, log_data: dict):
        """Delegates appending a log message to the LogWidget."""
        self.log_widget.append_log_message(log_data)

    def clear_all(self):
        """Clears all UI elements for a new experiment."""
        self.trial_view_cache.clear()
        for widget in self.visualization_widgets:
            if hasattr(widget, "clear"):
                widget.clear()
        self.insights_widget.clear()
        self.log_widget.clear_log()
        self.trials_table_widget.clear()

    def _on_insight_selected(self, item: "InsightListItem"):
        """Handles insight selection and highlights the plot and table."""
        self.trials_table_widget.clear_highlights()

        selected_item = self.insights_widget.get_selected_item()
        if not item or not selected_item or item != selected_item:
            for widget in self.visualization_widgets:
                if hasattr(widget, "update_plot_highlight"):
                    widget.update_plot_highlight(set())
            self.insights_widget.clear_selection()
            return

        highlight_ids = set(item.insight.trial_ids)
        for widget in self.visualization_widgets:
            if hasattr(widget, "update_plot_highlight"):
                widget.update_plot_highlight(highlight_ids)
        highlight_color = QColor("#FFFACD")  # LemonChiffon
        self.trials_table_widget.highlight_rows(highlight_ids, highlight_color)

    def load_visualization_plugins(self):
        """Loads all visualization plugins."""
        # For now, we just hardcode the plugins.
        # In the future, we could discover them dynamically.
        self.visualization_plugins.append(PerformancePlotPlugin())
        self.visualization_plugins.append(HyperparameterPCAPlugin())
