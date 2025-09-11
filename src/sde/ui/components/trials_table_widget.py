from typing import Dict, Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QStyle
)

from ..models import UITrial
from ..view_model import ExperimentViewModel


class NumericTableWidgetItem(QTableWidgetItem):
    """A custom QTableWidgetItem that implements numeric sorting."""

    def __lt__(self, other):
        try:
            self_float = float(self.text())
            other_float = float(other.text())
            return self_float < other_float
        except (ValueError, TypeError):
            return super().__lt__(other)


class TrialsTableWidget(QWidget):
    """A widget for displaying and managing trials in a table."""

    trial_selected = pyqtSignal(str)
    trial_double_clicked = pyqtSignal(str)
    prune_trial_requested = pyqtSignal(str)
    prioritize_trial_requested = pyqtSignal(str)
    spawn_trial_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.trial_row_map: Dict[str, int] = {}
        self.view_model: Optional[ExperimentViewModel] = None
        self.star_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogApplyButton
        )

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)

        self.filter_status_combo = QComboBox()
        self.filter_status_combo.addItems(
            ["All Statuses", "ACTIVE", "PRUNED", "COMPLETED", "PENDING"]
        )
        self.filter_status_combo.setToolTip("Filter trials by their status.")

        self.filter_text_input = QLineEdit()
        self.filter_text_input.setPlaceholderText(
            "Filter by Trial ID or Algorithm Name..."
        )
        self.filter_text_input.setClearButtonEnabled(True)

        filter_layout.addWidget(QLabel("Filter by:"))
        filter_layout.addWidget(self.filter_status_combo)
        filter_layout.addWidget(self.filter_text_input, 1)

        self.trials_table = QTableWidget()
        self.setup_table()

        layout.addWidget(filter_widget)
        layout.addWidget(self.trials_table)

    def _connect_signals(self):
        self.trials_table.itemSelectionChanged.connect(self._on_trial_selection_changed)
        self.trials_table.itemDoubleClicked.connect(self._on_trial_double_clicked)
        self.trials_table.customContextMenuRequested.connect(self._show_trial_context_menu)
        self.filter_status_combo.currentIndexChanged.connect(self._update_trial_filter)
        self.filter_text_input.textChanged.connect(self._update_trial_filter)

    def set_view_model(self, view_model: ExperimentViewModel):
        self.view_model = view_model

    def setup_table(self):
        self.trials_table.setColumnCount(7)
        self.trials_table.setHorizontalHeaderLabels(
            [
                "Trial ID", "Algorithm", "Status", "Epoch",
                "Accuracy", "Loss", "Est. Time/Epoch",
            ]
        )
        self.trials_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.trials_table.setToolTip(
            "Double-click a row to view its hyperparameters.\n"
            "Right-click for more options."
        )
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
        self.trials_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def update_table(self, trials: Dict[str, UITrial], metric_name: str, trial_view_cache: Dict[str, UITrial]):
        current_trial_ids = set(trials.keys())
        cached_trial_ids = set(trial_view_cache.keys())

        for trial_id in cached_trial_ids - current_trial_ids:
            if trial_id in self.trial_row_map:
                self.trials_table.removeRow(self.trial_row_map.pop(trial_id))
            del trial_view_cache[trial_id]

        for trial_id, ui_trial in trials.items():
            if ui_trial != trial_view_cache.get(trial_id):
                self._update_trial_ui(ui_trial, metric_name)
                trial_view_cache[trial_id] = ui_trial

        self._update_trial_filter()

    def _update_trial_ui(self, ui_trial: UITrial, metric_name: str):
        trial_id = ui_trial.id
        hparam_tooltip = "<b>Hyperparameters:</b><br>" + "<br>".join(
            f"<b>{k}:</b> {v}" for k, v in ui_trial.hyperparameters.items()
        )

        if trial_id not in self.trial_row_map:
            row_position = self.trials_table.rowCount()
            self.trials_table.insertRow(row_position)
            self.trial_row_map[trial_id] = row_position

        row = self.trial_row_map[trial_id]
        background_color = ui_trial.row_background_color

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

    def _update_trial_filter(self):
        status_filter = self.filter_status_combo.currentText()
        text_filter = self.filter_text_input.text().lower()

        for row in range(self.trials_table.rowCount()):
            trial_id_item = self.trials_table.item(row, 0)
            algorithm_item = self.trials_table.item(row, 1)
            status_item = self.trials_table.item(row, 2)

            if not all([trial_id_item, algorithm_item, status_item]):
                continue

            status_match = (
                status_filter == "All Statuses" or status_item.text() == status_filter
            )
            text_match = (
                text_filter in trial_id_item.text().lower()
                or text_filter in algorithm_item.text().lower()
            )
            self.trials_table.setRowHidden(row, not (status_match and text_match))

    def _show_trial_context_menu(self, pos):
        trial_id = self._get_selected_trial_id()
        if not trial_id or not self.view_model:
            return

        menu = QMenu()
        prune_action = menu.addAction("Prune Trial")
        prioritize_action = menu.addAction("Increase Priority")
        spawn_action = menu.addAction("Spawn Similar Trial...")

        trial = self.view_model.trials.get(trial_id)
        if trial and trial.status not in ["ACTIVE", "PENDING"]:
            prune_action.setEnabled(False)
            prioritize_action.setEnabled(False)

        action = menu.exec(self.trials_table.mapToGlobal(pos))

        if action == prune_action:
            self.prune_trial_requested.emit(trial_id)
        elif action == prioritize_action:
            self.prioritize_trial_requested.emit(trial_id)
        elif action == spawn_action:
            self.spawn_trial_requested.emit(trial_id)

    def _get_selected_trial_id(self) -> Optional[str]:
        selected_items = self.trials_table.selectedItems()
        if not selected_items:
            return None
        row = selected_items[0].row()
        for tid, r in self.trial_row_map.items():
            if r == row:
                return tid
        return None

    def _on_trial_selection_changed(self):
        trial_id = self._get_selected_trial_id()
        self.trial_selected.emit(trial_id if trial_id else "")

    def _on_trial_double_clicked(self, item: QTableWidgetItem):
        row = item.row()
        for tid, r in self.trial_row_map.items():
            if r == row:
                self.trial_double_clicked.emit(tid)
                break

    def clear(self):
        self.trials_table.setRowCount(0)
        self.trial_row_map.clear()

    def highlight_rows(self, trial_ids: set, color: "QColor"):
        for row in range(self.trials_table.rowCount()):
            trial_id_item = self.trials_table.item(row, 0)
            if trial_id_item and trial_id_item.text() in trial_ids:
                for col in range(self.trials_table.columnCount()):
                    self.trials_table.item(row, col).setBackground(color)

    def clear_highlights(self):
        if not self.view_model:
            return
        for row in range(self.trials_table.rowCount()):
            trial_id_item = self.trials_table.item(row, 0)
            if trial_id_item:
                ui_trial = self.view_model.trials.get(trial_id_item.text())
                if ui_trial:
                    for col in range(self.trials_table.columnCount()):
                        self.trials_table.item(row, col).setBackground(
                            ui_trial.row_background_color
                        )
