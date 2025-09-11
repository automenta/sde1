from datetime import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import Union

import pyqtgraph as pg
from PyQt6.QtCore import QPropertyAnimation
from PyQt6.QtCore import Qt
from PyQt6.QtCore import Property  # type: ignore
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QAbstractItemView
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QGroupBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QHeaderView
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QListWidget
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtWidgets import QMenu
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QSplitter
from PyQt6.QtWidgets import QTableWidget
from PyQt6.QtWidgets import QTableWidgetItem
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from .models import UIInsight
from .view_model import ExperimentViewModel


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


class NumericTableWidgetItem(QTableWidgetItem):
    """A custom QTableWidgetItem that implements numeric sorting."""
    def __lt__(self, other):
        # Try to convert text to float for numeric comparison
        try:
            self_float = float(self.text())
            other_float = float(other.text())
            return self_float < other_float
        except (ValueError, TypeError):
            # Fallback to string comparison if conversion fails
            return super().__lt__(other)


class ResultsPane(QWidget):
    """The right-hand pane for displaying experiment results, including the plot,
    trials table, insights, and log.
    """

    # Signals for user interactions that the parent window needs to handle
    trial_selected = pyqtSignal(str)  # Emits trial_id
    trial_double_clicked = pyqtSignal(str)  # Emits trial_id
    insight_selected = pyqtSignal(set)  # Emits a set of trial_ids to highlight
    prune_trial_requested = pyqtSignal(str)
    prioritize_trial_requested = pyqtSignal(str)
    spawn_trial_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()


    def __init__(self, parent=None):
        super().__init__(parent)

        # --- UI State and Data Maps ---
        self.trial_row_map: Dict[str, int] = {}
        self.plot_curve_map: Dict[str, pg.PlotDataItem] = {}
        self.legend: Optional[pg.LegendItem] = None
        self.selected_insight_item: Optional[InsightListItem] = None
        self.displayed_insight_count = 0
        self.view_model: Optional[ExperimentViewModel] = None
        self.insight_animation: Optional[QPropertyAnimation] = None
        self.available_metrics: Set[str] = set()
        self.plot_curve_visibility: Dict[str, bool] = {}
        self.star_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogApplyButton
        )

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initializes the main UI layout and sub-components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Main Content Splitter ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # --- Plot Widget ---
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.setup_plot()

        # Metric selection UI
        metric_selection_layout = QHBoxLayout()
        metric_selection_layout.addStretch()
        metric_label = QLabel("<b>Plotting Metric:</b>")
        metric_selection_layout.addWidget(metric_label)
        self.metric_combo = QComboBox()
        self.metric_combo.setMinimumWidth(150)
        metric_selection_layout.addWidget(self.metric_combo)

        plot_layout.addLayout(metric_selection_layout)
        plot_layout.addWidget(self.plot_widget)


        # --- Bottom Pane (Table, Insights, Log) ---
        bottom_pane = QWidget()
        bottom_layout = QVBoxLayout(bottom_pane) # Changed to QVBoxLayout

        # --- Filter Controls ---
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)

        self.filter_status_combo = QComboBox()
        self.filter_status_combo.addItems(["All Statuses", "ACTIVE", "PRUNED", "COMPLETED", "PENDING"])
        self.filter_status_combo.setToolTip("Filter trials by their status.")

        self.filter_text_input = QLineEdit()
        self.filter_text_input.setPlaceholderText("Filter by Trial ID or Algorithm Name...")
        self.filter_text_input.setClearButtonEnabled(True)

        filter_layout.addWidget(QLabel("Filter by:"))
        filter_layout.addWidget(self.filter_status_combo)
        filter_layout.addWidget(self.filter_text_input, 1) # Stretch the text input

        # --- Trials Table ---
        self.trials_table = QTableWidget()
        self.setup_table()

        # Add filter controls and table to a container
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.addWidget(filter_widget)
        table_layout.addWidget(self.trials_table)
        table_layout.setContentsMargins(0, 0, 0, 0)


        right_bottom_splitter = QSplitter(Qt.Orientation.Vertical)

        # Insights Group
        self.insights_group = QGroupBox("Insights")
        insights_layout = QVBoxLayout(self.insights_group)
        self.insights_list = QListWidget()
        self.insights_list.setWordWrap(True)
        insights_layout.addWidget(self.insights_list)
        insights_layout.setContentsMargins(0, 5, 0, 0)
        self.insights_group.setLayout(insights_layout)

        # Log Group
        log_group = QGroupBox("Log")
        log_outer_layout = QVBoxLayout()
        log_group.setLayout(log_outer_layout)

        log_header_layout = QHBoxLayout()
        log_header_layout.addStretch()
        self.refresh_button = QPushButton("Refresh")
        clear_log_button = QPushButton("Clear Log")
        log_header_layout.addWidget(self.refresh_button)
        log_header_layout.addWidget(clear_log_button)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        log_outer_layout.addLayout(log_header_layout)
        log_outer_layout.addWidget(self.log_text_edit)
        log_outer_layout.setContentsMargins(0, 5, 0, 0)

        self.clear_log_button = clear_log_button

        right_bottom_splitter.addWidget(self.insights_group)
        right_bottom_splitter.addWidget(log_group)
        right_bottom_splitter.setSizes([100, 200])

        # Bottom Splitter
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(table_container)
        bottom_splitter.addWidget(right_bottom_splitter)
        bottom_splitter.setSizes([750, 450])
        bottom_layout.addWidget(bottom_splitter)
        # This was incorrect, bottom_layout is already on bottom_pane
        # bottom_pane.setLayout(bottom_layout)

        splitter.addWidget(plot_container)
        splitter.addWidget(bottom_pane)
        splitter.setSizes([500, 300])

    def _connect_signals(self):
        """Connects internal widget signals to the pane's public signals."""
        self.trials_table.itemSelectionChanged.connect(self._on_trial_selection_changed)
        self.trials_table.itemDoubleClicked.connect(self._on_trial_double_clicked)
        self.insights_list.itemClicked.connect(self._on_insight_selected)
        self.clear_log_button.clicked.connect(self.clear_log)
        self.refresh_button.clicked.connect(self.refresh_requested)
        self.trials_table.customContextMenuRequested.connect(self._show_trial_context_menu)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        self.filter_status_combo.currentIndexChanged.connect(self._update_trial_filter)
        self.filter_text_input.textChanged.connect(self._update_trial_filter)
        if self.legend:
            self.legend.itemClicked.connect(self._on_legend_item_clicked)


    # --- Public Methods for Updating the View ---

    def update_view(self, view_model: ExperimentViewModel):
        """The main entry point for refreshing the entire results view."""
        self.view_model = view_model
        self.update_trials_and_plots(view_model)
        self.update_insights_list(view_model)

    def setup_plot(self):
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("Real-Time Trial Performance", color="k", size="16pt")
        self.plot_widget.setLabel(
            "left", "Accuracy", color="k", **{"font-size": "12pt"}
        )
        self.plot_widget.setLabel("bottom", "Epoch", color="k", **{"font-size": "12pt"})
        self.plot_widget.showGrid(x=True, y=True)
        self.legend = self.plot_widget.addLegend()
        self.plot_curve_visibility = {} # trial_id -> bool

    def setup_table(self):
        self.trials_table.setColumnCount(7)
        self.trials_table.setHorizontalHeaderLabels(
            [
                "Trial ID",
                "Algorithm",
                "Status",
                "Epoch",
                "Accuracy",
                "Loss",
                "Est. Time/Epoch",
            ]
        )
        self.trials_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.trials_table.setToolTip("Double-click a row to view its hyperparameters.\nRight-click for more options.")
        self.trials_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header = self.trials_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        self.trials_table.setSortingEnabled(True)
        self.trials_table.setColumnWidth(0, 100)
        self.trials_table.setColumnWidth(1, 120)
        self.trials_table.setColumnWidth(2, 100)
        self.trials_table.setColumnWidth(3, 60)
        self.trials_table.setColumnWidth(4, 100)
        self.trials_table.setColumnWidth(5, 100)
        self.trials_table.setColumnWidth(6, 120)
        self.trials_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.trials_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

    def _update_trial_filter(self):
        """Filters the trials table based on the status combo box and text input."""
        status_filter = self.filter_status_combo.currentText()
        text_filter = self.filter_text_input.text().lower()

        for row in range(self.trials_table.rowCount()):
            # Column indices: 0 = Trial ID, 1 = Algorithm, 2 = Status
            trial_id_item = self.trials_table.item(row, 0)
            algorithm_item = self.trials_table.item(row, 1)
            status_item = self.trials_table.item(row, 2)

            if not all([trial_id_item, algorithm_item, status_item]):
                continue

            # Check status filter
            status_match = (status_filter == "All Statuses" or status_item.text() == status_filter)

            # Check text filter
            text_match = (
                text_filter in trial_id_item.text().lower() or
                text_filter in algorithm_item.text().lower()
            )

            # Show or hide the row
            self.trials_table.setRowHidden(row, not (status_match and text_match))

    def update_trials_and_plots(self, view_model: ExperimentViewModel):
        """Updates the trials table and plot widget from the ViewModel."""
        # Use the combo box's current selection as the metric to display
        metric_name = self.metric_combo.currentText() or view_model.performance_metric_name
        self._update_available_metrics(view_model)


        current_trial_ids = set(view_model.trials.keys())
        existing_ui_trial_ids = set(self.trial_row_map.keys())

        for trial_id in existing_ui_trial_ids - current_trial_ids:
            row = self.trial_row_map.pop(trial_id)
            self.trials_table.removeRow(row)
            if trial_id in self.plot_curve_map:
                self.plot_widget.removeItem(self.plot_curve_map.pop(trial_id))

        for trial_id, ui_trial in view_model.trials.items():
            self._update_trial_ui(ui_trial, metric_name)
            if trial_id in self.plot_curve_map:
                metric_list = ui_trial.results.get(metric_name, [])
                if metric_list:
                    try:
                        epochs, metrics = zip(*metric_list)
                        self.plot_curve_map[trial_id].setData(epochs, metrics)
                    except ValueError:
                        self.plot_curve_map[trial_id].clear()

        self._update_trial_filter()

    def _update_trial_ui(self, ui_trial, metric_name: str):
        """Updates or creates a row in the trials table for a given UITrial."""
        trial_id = ui_trial.id

        # Create a rich HTML tooltip with all hyperparameters
        hparam_tooltip = "<b>Hyperparameters:</b><br>" + "<br>".join(
            f"<b>{k}:</b> {v}" for k, v in ui_trial.hyperparameters.items()
        )

        if trial_id not in self.trial_row_map:
            row_position = self.trials_table.rowCount()
            self.trials_table.insertRow(row_position)
            self.trial_row_map[trial_id] = row_position
            self.plot_curve_visibility[trial_id] = True # Default to visible

            name = f"{ui_trial.algorithm_name} ({trial_id[:6]})"
            pen = ui_trial.pen
            self.plot_curve_map[trial_id] = self.plot_widget.plot(
                [], [], name=name, pen=pen, symbol="o", symbolSize=6, symbolBrush=pen.color()
            )

        row = self.trial_row_map[trial_id]
        background_color = ui_trial.row_background_color
        self._apply_plot_curve_styles()

        # Use a mix of regular and numeric items for appropriate sorting
        trial_id_item = QTableWidgetItem(trial_id)
        if ui_trial.prioritized:
            trial_id_item.setIcon(self.star_icon)

        items = [
            trial_id_item,
            QTableWidgetItem(ui_trial.algorithm_name),
            QTableWidgetItem(ui_trial.status),
            NumericTableWidgetItem(ui_trial.display_epoch),
            NumericTableWidgetItem(ui_trial.get_latest_metric(metric_name)),
            NumericTableWidgetItem(ui_trial.get_latest_metric("loss")),
            QTableWidgetItem(ui_trial.display_est_time),
        ]

        for col, item in enumerate(items):
            item.setBackground(background_color)
            item.setToolTip(hparam_tooltip)
            self.trials_table.setItem(row, col, item)

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

        # Trigger animation only if there are new insights
        if num_new_insights > 0:
            self._trigger_insight_animation()

    def _trigger_insight_animation(self):
        """Animates the border of the 'Insights' group box to signal a new insight."""
        if self.insight_animation and self.insight_animation.state() == QPropertyAnimation.State.Running:
            return  # Don't start a new animation if one is already running

        self.insight_animation = QPropertyAnimation(self, b"insightBorderColor")
        self.insight_animation.setDuration(1500)
        self.insight_animation.setStartValue(QColor("#0078D7"))  # Start with highlight color
        self.insight_animation.setEndValue(QColor("lightgray"))  # End with default color
        self.insight_animation.setEasingCurve(Qt.EasingCurve.OutCubic)
        self.insight_animation.start()

    # This is a custom property setter required for QPropertyAnimation to work on a non-standard property
    def _set_insight_border_color(self, color: QColor):
        """Sets the border color of the insights group box."""
        self.insights_group.setStyleSheet(f"QGroupBox {{ border: 1px solid {color.name()}; margin-top: 1em; }}")

    # This registers the custom property with Qt's meta-object system
    insightBorderColor = Property(QColor, fset=_set_insight_border_color)  # type: ignore

    def update_plot_highlight(
        self, highlight_ids: set, view_model: ExperimentViewModel
    ):
        """Highlights a specific set of trials on the plot by re-applying all styles."""
        # The new logic is now centralized in _apply_plot_curve_styles.
        # We just need to trigger it. The highlight_ids are read from self.selected_insight_item.
        self._apply_plot_curve_styles()

    def append_log_message(self, log_data: dict):
        """Appends a structured log message to the text edit with color-coding."""
        level = log_data.get("level", "INFO").upper()
        message = log_data.get("message", "")

        color_map = {
            "INFO": "#000000",      # Black
            "WARN": "#FFA500",      # Orange
            "ERROR": "#DC143C",     # Crimson
            "INSIGHT": "#8A2BE2",   # BlueViolet
        }
        color = color_map.get(level, "black")

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = (
            f'<span style="color: #808080;">[{timestamp}]</span> '
            f'<b style="color: {color};">[{level}]</b> '
            f'<span style="color: #36454F;">{message}</span>'
        )
        self.log_text_edit.append(formatted_message)
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )

    def clear_log(self):
        """Clears the log text edit."""
        self.log_text_edit.clear()
        self.append_log_message({"level": "INFO", "message": "Log cleared."})

    def clear_all(self):
        """Clears all UI elements for a new experiment."""
        self.trials_table.setRowCount(0)
        self.plot_widget.clear()
        self.trial_row_map.clear()
        self.plot_curve_map.clear()
        self.metric_combo.clear()
        self.available_metrics.clear()
        self.insights_list.clear()
        self.insights_group.setStyleSheet("")  # Reset stylesheet
        self.displayed_insight_count = 0
        self.setup_plot()  # Re-add legend and titles

    # --- Internal Signal Handlers ---

    def _on_trial_selection_changed(self):
        """Emits the ID of the selected trial."""
        selected_items = self.trials_table.selectedItems()
        if not selected_items:
            self.trial_selected.emit("")  # Emit empty string to clear selection
            return

        selected_row = self.trials_table.currentRow()
        for tid, r in self.trial_row_map.items():
            if r == selected_row:
                self.trial_selected.emit(tid)
                break

    def _on_trial_double_clicked(self, item: QTableWidgetItem):
        """Emits the ID of the double-clicked trial."""
        row = item.row()
        for tid, r in self.trial_row_map.items():
            if r == row:
                self.trial_double_clicked.emit(tid)
                break

    def _on_insight_selected(self, item: InsightListItem):
        """Handles insight selection and emits the relevant trial IDs."""
        if not isinstance(item, InsightListItem):
            return

        if self.selected_insight_item == item:
            self.insights_list.clearSelection()
            self.selected_insight_item = None
            self.insight_selected.emit(set())  # Emit empty set to clear highlights
            return

        self.selected_insight_item = item
        highlight_ids = set(item.insight.trial_ids)
        self.insight_selected.emit(highlight_ids)

        # --- Insight Focus Mode ---
        if self.selected_insight_item:
            # When an insight is selected, filter the table to show only relevant trials
            short_ids = [tid[:8] for tid in highlight_ids]
            filter_text = "|".join(short_ids)
            self.filter_text_input.setText(filter_text)
            self.filter_status_combo.setCurrentText("All Statuses")
        else:
            # When selection is cleared, clear the filter
            self.filter_text_input.clear()

        # Also select the rows in the table
        self.trials_table.clearSelection()
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for row in range(self.trials_table.rowCount()):
            # Check if row is visible before selecting
            if not self.trials_table.isRowHidden(row):
                trial_id_item = self.trials_table.item(row, 0)
                if trial_id_item and any(tid.startswith(trial_id_item.text()) for tid in highlight_ids):
                     self.trials_table.selectRow(row)
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def _on_legend_item_clicked(self, curve_item, label_item):
        """Toggles the visibility of a plot curve when its legend item is clicked."""
        # Find the trial_id associated with the clicked curve
        clicked_trial_id = None
        for trial_id, curve in self.plot_curve_map.items():
            if curve is curve_item:
                clicked_trial_id = trial_id
                break

        if clicked_trial_id:
            # Toggle visibility state
            self.plot_curve_visibility[clicked_trial_id] = not self.plot_curve_visibility.get(clicked_trial_id, True)
            self._apply_plot_curve_styles()

    def _apply_plot_curve_styles(self):
        """Applies visibility and highlight styles to all plot curves."""
        if not self.view_model:
            return

        # Get the set of highlighted trials from the currently selected insight, if any
        highlight_ids = set()
        if self.selected_insight_item:
            highlight_ids = set(self.selected_insight_item.insight.trial_ids)

        for trial_id, curve in self.plot_curve_map.items():
            ui_trial = self.view_model.trials.get(trial_id)
            if not ui_trial:
                continue

            pen = ui_trial.pen
            color = pen.color()
            is_visible = self.plot_curve_visibility.get(trial_id, True)
            is_highlighted = trial_id in highlight_ids

            if is_highlighted:
                color.setAlpha(255)
                curve.setPen(pg.mkPen(color=color, width=4))
                curve.setZValue(100)
            elif is_visible:
                color.setAlpha(200) # Slightly less opaque than highlighted
                curve.setPen(pg.mkPen(color=color, width=2))
                curve.setZValue(0)
            else:
                color.setAlpha(15) # Barely visible
                curve.setPen(pg.mkPen(color=color, width=1, style=Qt.PenStyle.DotLine))
                curve.setZValue(-100)

            # Update legend label color
            for _, label in self.legend.items:
                if label.text == curve.name():
                    label.setText(label.text, color='k' if is_visible else 'gray')
                    break


    def _on_metric_changed(self):
        """Handles the metric selection change by replotting all data."""
        if not self.view_model:
            return

        metric_name = self.metric_combo.currentText()
        if not metric_name:
            return

        # Update plot labels
        self.plot_widget.setLabel("left", metric_name.replace("_", " ").title())
        self.plot_widget.setTitle(f"Real-Time Trial Performance: {metric_name.replace('_', ' ').title()}", color="k", size="16pt")


        # Update plot data for all existing curves
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

    def _update_available_metrics(self, view_model: ExperimentViewModel):
        """Discovers and populates the metric combo box from trial data."""
        new_metrics: Set[str] = set()
        for trial in view_model.trials.values():
            new_metrics.update(trial.results.keys())

        if new_metrics != self.available_metrics:
            self.available_metrics = new_metrics
            current_selection = self.metric_combo.currentText()
            self.metric_combo.blockSignals(True)
            self.metric_combo.clear()
            sorted_metrics = sorted(list(self.available_metrics))
            if sorted_metrics:
                self.metric_combo.addItems(sorted_metrics)
                # Try to restore previous selection
                if current_selection in sorted_metrics:
                    self.metric_combo.setCurrentText(current_selection)
                # Or set a sensible default
                elif view_model.performance_metric_name in sorted_metrics:
                    self.metric_combo.setCurrentText(view_model.performance_metric_name)
            self.metric_combo.blockSignals(False)

    def _show_trial_context_menu(self, pos):
        """Creates and shows a context menu for a trial."""
        trial_id = self._get_selected_trial_id()
        if not trial_id or not self.view_model:
            return

        menu = QMenu()
        prune_action = menu.addAction("Prune Trial")
        prioritize_action = menu.addAction("Increase Priority")
        spawn_action = menu.addAction("Spawn Similar Trial...")

        # Disable actions based on trial status if needed
        trial = self.view_model.trials.get(trial_id)
        if trial and trial.status not in ["ACTIVE", "PENDING"]:
            prune_action.setEnabled(False)
            prioritize_action.setEnabled(False)

        action = menu.exec(self.trials_table.mapToGlobal(pos))

        if action == prune_action:
            self._prune_selected_trial()
        elif action == prioritize_action:
            self._prioritize_selected_trial()
        elif action == spawn_action:
            self._spawn_similar_trial()

    def _get_selected_trial_id(self) -> Optional[str]:
        """Helper to get the ID of the currently selected trial."""
        selected_items = self.trials_table.selectedItems()
        if not selected_items:
            return None

        row = selected_items[0].row()
        for tid, r in self.trial_row_map.items():
            if r == row:
                return tid
        return None

    def _prune_selected_trial(self):
        trial_id = self._get_selected_trial_id()
        if trial_id:
            self.prune_trial_requested.emit(trial_id)

    def _prioritize_selected_trial(self):
        trial_id = self._get_selected_trial_id()
        if trial_id:
            self.prioritize_trial_requested.emit(trial_id)

    def _spawn_similar_trial(self):
        trial_id = self._get_selected_trial_id()
        if trial_id:
            self.spawn_trial_requested.emit(trial_id)
