from datetime import datetime

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView
from PyQt6.QtWidgets import QGroupBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QHeaderView
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

    def __init__(self, ui_insight: UIInsight, parent: QListWidget | None = None):
        super().__init__(parent)
        self.insight = ui_insight
        self.setIcon(ui_insight.icon)
        self.setText(f"[{ui_insight.timestamp}] {ui_insight.message}")
        self.setToolTip(ui_insight.message)


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


    def __init__(self, parent=None):
        super().__init__(parent)

        # --- UI State and Data Maps ---
        self.trial_row_map = {}  # trial.id -> table_row_index
        self.plot_curve_map = {}  # trial.id -> plot_curve_item
        self.legend = None
        self.selected_insight_item = None
        self.displayed_insight_count = 0
        self.view_model = None # To access trial data in context menu

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
        self.plot_widget = pg.PlotWidget()
        self.setup_plot()

        # --- Bottom Pane (Table, Insights, Log) ---
        bottom_pane = QWidget()
        bottom_layout = QHBoxLayout(bottom_pane)

        self.trials_table = QTableWidget()
        self.setup_table()

        right_bottom_splitter = QSplitter(Qt.Orientation.Vertical)

        # Insights Group
        insights_group = QGroupBox("Insights")
        insights_layout = QVBoxLayout(insights_group)
        self.insights_list = QListWidget()
        self.insights_list.setWordWrap(True)
        insights_layout.addWidget(self.insights_list)
        insights_layout.setContentsMargins(0, 5, 0, 0)
        insights_group.setLayout(insights_layout)

        # Log Group
        log_group = QGroupBox("Log")
        log_outer_layout = QVBoxLayout()
        log_group.setLayout(log_outer_layout)

        log_header_layout = QHBoxLayout()
        log_header_layout.addStretch()
        clear_log_button = QPushButton("Clear Log")
        log_header_layout.addWidget(clear_log_button)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        log_outer_layout.addLayout(log_header_layout)
        log_outer_layout.addWidget(self.log_text_edit)
        log_outer_layout.setContentsMargins(0, 5, 0, 0)

        self.clear_log_button = clear_log_button

        right_bottom_splitter.addWidget(insights_group)
        right_bottom_splitter.addWidget(log_group)
        right_bottom_splitter.setSizes([100, 200])

        # Bottom Splitter
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self.trials_table)
        bottom_splitter.addWidget(right_bottom_splitter)
        bottom_splitter.setSizes([750, 450])
        bottom_layout.addWidget(bottom_splitter)
        bottom_pane.setLayout(bottom_layout)

        splitter.addWidget(self.plot_widget)
        splitter.addWidget(bottom_pane)
        splitter.setSizes([500, 300])

    def _connect_signals(self):
        """Connects internal widget signals to the pane's public signals."""
        self.trials_table.itemSelectionChanged.connect(self._on_trial_selection_changed)
        self.trials_table.itemDoubleClicked.connect(self._on_trial_double_clicked)
        self.insights_list.itemClicked.connect(self._on_insight_selected)
        self.clear_log_button.clicked.connect(self.clear_log)
        self.trials_table.customContextMenuRequested.connect(self._show_trial_context_menu)


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

    def update_trials_and_plots(self, view_model: ExperimentViewModel):
        """Updates the trials table and plot widget from the ViewModel."""
        metric_name = view_model.performance_metric_name

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

    def _update_trial_ui(self, ui_trial, metric_name: str):
        """Updates or creates a row in the trials table for a given UITrial."""
        trial_id = ui_trial.id
        if trial_id not in self.trial_row_map:
            row_position = self.trials_table.rowCount()
            self.trials_table.insertRow(row_position)
            self.trial_row_map[trial_id] = row_position

            name = f"{ui_trial.algorithm_name} ({trial_id[:6]})"
            pen = ui_trial.pen
            self.plot_curve_map[trial_id] = self.plot_widget.plot(
                [], [], name=name, pen=pen, symbol="o", symbolSize=6, symbolBrush=pen.color()
            )

        row = self.trial_row_map[trial_id]
        self.plot_curve_map[trial_id].setPen(ui_trial.pen)
        background_color = ui_trial.row_background_color

        self.trials_table.setItem(row, 0, QTableWidgetItem(trial_id))
        self.trials_table.setItem(row, 1, QTableWidgetItem(ui_trial.algorithm_name))
        self.trials_table.setItem(row, 2, QTableWidgetItem(ui_trial.status))
        self.trials_table.setItem(row, 3, QTableWidgetItem(ui_trial.display_epoch))
        self.trials_table.setItem(row, 4, QTableWidgetItem(ui_trial.get_latest_metric(metric_name)))
        self.trials_table.setItem(row, 5, QTableWidgetItem(ui_trial.get_latest_metric("loss")))
        self.trials_table.setItem(row, 6, QTableWidgetItem(ui_trial.display_est_time))

        for col in range(self.trials_table.columnCount()):
            item = self.trials_table.item(row, col)
            if not item:
                item = QTableWidgetItem()
                self.trials_table.setItem(row, col, item)
            item.setBackground(background_color)

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

    def update_plot_highlight(self, highlight_ids: set, view_model: ExperimentViewModel):
        """Highlights a specific set of trials on the plot."""
        for trial_id, curve in self.plot_curve_map.items():
            ui_trial = view_model.trials.get(trial_id)
            if not ui_trial:
                continue

            pen = ui_trial.pen
            color = pen.color()

            if trial_id in highlight_ids:
                color.setAlpha(255)
                curve.setPen(pg.mkPen(color=color, width=4))
                curve.setZValue(100)
            else:
                color.setAlpha(30)
                curve.setPen(pg.mkPen(color=color, width=1))
                curve.setZValue(0)

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
        self.insights_list.clear()
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

        # Also select the rows in the table
        self.trials_table.clearSelection()
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for trial_id, row in self.trial_row_map.items():
            if trial_id in highlight_ids:
                self.trials_table.selectRow(row)
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

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

    def _get_selected_trial_id(self) -> str | None:
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
