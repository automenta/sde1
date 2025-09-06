import sys
import json
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTextEdit,
    QSplitter,
    QPushButton,
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFormLayout,
    QSlider,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QMessageBox,
    QDialog,
    QStyle,
    QAbstractItemView,
    QTreeWidget,
    QTreeWidgetItem,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Import backend and UI components
from sde.core.actions import ActionType
from sde.engine.orchestrator import ExperimentOrchestrator
from sde.models import AVAILABLE_MODELS
from sde.challenges import AVAILABLE_DATASETS
from sde.ui.hyperparameters import HyperparameterDialog
from sde.ui.view_model import ExperimentViewModel
from sde.ui.models import UIInsight
from sde.ui.setup_pane import SetupPane


class InsightListItem(QListWidgetItem):
    """A custom QListWidgetItem that stores the full UIInsight object."""

    def __init__(self, ui_insight: UIInsight, parent: QListWidget | None = None):
        super().__init__(parent)
        self.insight = ui_insight
        self.setIcon(ui_insight.icon)
        self.setText(f"[{ui_insight.timestamp}] {ui_insight.message}")
        self.setToolTip(ui_insight.message)


class HyperparameterViewerDialog(QDialog):
    """A dialog to display nested hyperparameter dictionaries in a readable tree."""

    def __init__(self, hparams: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hyperparameter Details")
        self.setLayout(QVBoxLayout())
        self.resize(450, 350)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Parameter", "Value"])
        self.layout().addWidget(self.tree)

        self.populate_tree(hparams)
        self.tree.expandAll()
        for i in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        self.layout().addWidget(ok_button)

    def populate_tree(self, data: dict, parent_item: QTreeWidgetItem = None):
        if parent_item is None:
            parent_item = self.tree.invisibleRootItem()

        for key, value in data.items():
            item = QTreeWidgetItem(parent_item, [str(key)])
            if isinstance(value, dict):
                self.populate_tree(value, item)
            else:
                if isinstance(value, float):
                    value_str = f"{value:.6g}"  # Use general format for nice printing
                else:
                    value_str = str(value)
                item.setText(1, value_str)


class MainWindow(QMainWindow):
    """
    The main window for the Scientific Discovery Engine UI.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Discovery Engine")
        self.setGeometry(100, 100, 1400, 900)

        # --- Backend and ViewModel ---
        self.orchestrator = ExperimentOrchestrator()
        self.view_model = ExperimentViewModel(self.style())

        # --- UI State and Data Maps ---
        self.trial_row_map = {}  # trial.id -> table_row_index
        self.plot_curve_map = {}  # trial.id -> plot_curve_item
        self.legend = None
        self.selected_insight_item = None

        self._init_ui()
        self._connect_signals()

        # Populate initial data and states
        self.update_button_states() # Initialize button states
        self.append_log_message(
            "INFO: UI Initialized. Configure your experiment and click 'Start'."
        )

    def _init_ui(self):
        """Initializes the main UI layout and sub-components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.setup_pane = SetupPane()
        right_pane = self._create_right_pane()

        main_layout.addWidget(self.setup_pane)
        main_layout.addWidget(right_pane, 1)

    def _create_right_pane(self) -> QWidget:
        """Creates the right-hand pane for displaying results."""
        results_pane = QWidget()
        results_layout = QVBoxLayout(results_pane)

        # --- Main Content Splitter ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        results_layout.addWidget(splitter)

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
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        log_layout.setContentsMargins(0, 5, 0, 0)
        log_group.setLayout(log_layout)

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

        return results_pane

    def _connect_signals(self):
        """Connects all UI signals to their corresponding slots."""
        # Backend signals
        self.orchestrator.log_message.connect(self.append_log_message)
        self.orchestrator.state_changed.connect(self.on_state_changed)

        # Setup Pane signals
        self.setup_pane.dataset_changed.connect(self.update_model_list)
        self.setup_pane.start_simple_run_requested.connect(self.start_simple_experiment)
        self.setup_pane.tune_run_requested.connect(self.open_tuning_dialog)
        self.setup_pane.add_models_requested.connect(self.add_models_to_run)
        self.setup_pane.pause_run_requested.connect(self.pause_experiment)
        self.setup_pane.resume_run_requested.connect(self.resume_experiment)
        self.setup_pane.throttle_changed.connect(self.update_throttle)
        self.setup_pane.remove_algorithm_requested.connect(self.remove_algorithm)

        # Right pane controls
        self.trials_table.itemSelectionChanged.connect(self.on_trial_selected)
        self.insights_list.itemClicked.connect(self.on_insight_selected)

    def show_hyperparameter_dialog(self, hparams: dict):
        """Shows the hyperparameter viewer dialog for the given parameters."""
        if not hparams:
            QMessageBox.information(self, "Info", "No hyperparameters to display.")
            return
        dialog = HyperparameterViewerDialog(hparams, self)
        dialog.exec()

    def update_model_list(self, dataset_name: str):
        """Updates the model list in the setup pane."""
        self.setup_pane.update_model_list(dataset_name)

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
        self.trials_table.setColumnCount(8)
        self.trials_table.setHorizontalHeaderLabels(
            [
                "Trial ID",
                "Algorithm",
                "Status",
                "Epoch",
                "Accuracy",
                "Loss",
                "Est. Time/Epoch",
                "Hyperparameters",
            ]
        )
        header = self.trials_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
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

    def on_state_changed(self, state: dict):
        """
        The central handler for all state updates from the Orchestrator.
        It delegates state processing to the ViewModel and then triggers a UI refresh.
        """
        self.view_model.update_state(state)
        self._update_all_widgets()

    def _update_all_widgets(self):
        """Refreshes all UI components based on the current ViewModel state."""
        self.setup_pane.update_button_states(
            self.view_model.valid_actions,
            self.view_model.status,
            bool(self.view_model.challenge_name)
        )
        self.setup_pane.update_algorithm_table(
            self.view_model.algorithms, self.view_model.valid_actions
        )
        self.update_trials_and_plots()
        self.update_progress_bar()
        self.update_insights_list()

    def update_trials_and_plots(self):
        """Updates the trials table and plot widget from the ViewModel."""
        metric_name = self.view_model.performance_metric_name

        # Prune UI elements for trials that no longer exist
        current_trial_ids = set(self.view_model.trials.keys())
        existing_ui_trial_ids = set(self.trial_row_map.keys())

        for trial_id in existing_ui_trial_ids - current_trial_ids:
            row = self.trial_row_map.pop(trial_id)
            self.trials_table.removeRow(row)
            if trial_id in self.plot_curve_map:
                self.plot_widget.removeItem(self.plot_curve_map.pop(trial_id))

        # Update existing trials and add new ones
        for trial_id, ui_trial in self.view_model.trials.items():
            self.update_trial_ui(ui_trial, metric_name)

            # Update plot
            if trial_id in self.plot_curve_map:
                metric_list = ui_trial.results.get(metric_name, [])
                if metric_list:
                    try:
                        epochs, metrics = zip(*metric_list)
                        self.plot_curve_map[trial_id].setData(epochs, metrics)
                    except ValueError:
                        self.plot_curve_map[trial_id].clear()

    def update_progress_bar(self):
        """Updates the progress bar based on trial statuses in the ViewModel."""
        trials = self.view_model.trials.values()
        if not trials:
            self.setup_pane.update_progress_bar(0)
            return

        total_trials = len(trials)
        completed_statuses = {"COMPLETED", "PRUNED"}
        completed_trials = sum(1 for t in trials if t.status in completed_statuses)
        progress = int((completed_trials / total_trials) * 100)
        self.setup_pane.update_progress_bar(progress)

    def update_insights_list(self):
        """Updates the insights list from the ViewModel."""
        # A simple approach: clear and redraw. More complex logic could be used
        # to avoid flickering, but this is robust for now.
        self.insights_list.clear()
        for ui_insight in self.view_model.insights:
            item = InsightListItem(ui_insight, self.insights_list)
            self.insights_list.addItem(item)
        self.insights_list.scrollToBottom()

    def update_throttle(self, value: int):
        # Dispatch a SET_BUDGET action. This is a conceptual mapping for now.
        self.orchestrator.dispatch(ActionType.SET_BUDGET, {"worker_throttle_percent": value})

    def pause_experiment(self):
        self.orchestrator.dispatch(ActionType.PAUSE_RUN, {})

    def resume_experiment(self):
        self.orchestrator.dispatch(ActionType.RESUME_RUN, {})

    def remove_algorithm(self, algorithm_id: str):
        self.orchestrator.dispatch(ActionType.REMOVE_ALGORITHM, {"algorithm_id": algorithm_id})

    def start_simple_experiment(self, settings: dict):
        """
        Dispatches actions to the orchestrator to build and start an experiment.
        """
        dataset_name, selected_models = self.setup_pane.get_experiment_settings()
        if not dataset_name or not selected_models:
            self.append_log_message(
                "ERROR: Please select a dataset and at least one model."
            )
            return

        self.append_log_message(
            f"INFO: Configuring experiment on '{dataset_name}' with models: {selected_models}"
        )
        self._clear_previous_experiment()

        # --- Dispatch Actions ---
        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.orchestrator.dispatch(
            ActionType.SET_CHALLENGE, {"name": dataset_name, "type": challenge_def.type}
        )

        for model_name in selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = self._create_default_param_space(model_def)
            self.orchestrator.dispatch(
                ActionType.ADD_ALGORITHM, {"name": model_name, "parameter_space": param_space}
            )

        start_payload = {"enable_checkpointing": settings["enable_checkpointing"]}
        self.orchestrator.dispatch(ActionType.START_RUN, start_payload)

    def _create_default_param_space(self, model_def: dict) -> dict:
        """Creates a simplified, default parameter space for a given model."""
        return {
            k: (v["min"], v["max"])
            for param_type in model_def.hyperparameter_schema.values()
            for k, v in param_type.items()
        }

    def add_models_to_run(self):
        """Adds newly selected models to an already running experiment."""
        existing_algo_names = {algo.name for algo in self.view_model.algorithms.values()}
        newly_selected_models = self.setup_pane.get_newly_selected_models(existing_algo_names)

        if not newly_selected_models:
            self.append_log_message("INFO: No new models selected to add.")
            return

        self.append_log_message(f"INFO: Adding new models to run: {', '.join(newly_selected_models)}")
        for model_name in newly_selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = self._create_default_param_space(model_def)
            self.orchestrator.dispatch(ActionType.ADD_ALGORITHM, {"name": model_name, "parameter_space": param_space})

    def open_tuning_dialog(self):
        """Opens the tuning dialog and configures the experiment via the orchestrator."""
        dataset_name, selected_models_names = self.setup_pane.get_experiment_settings()
        if not dataset_name or not selected_models_names:
            QMessageBox.warning(self, "Warning", "Please select a dataset and at least one model to tune.")
            return

        selected_model_defs = [AVAILABLE_MODELS[name] for name in selected_models_names]
        dialog = HyperparameterDialog(selected_model_defs, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_configuration()
        self.append_log_message(f"INFO: Configuring tuning experiment with scheduler '{config['adaptive_scheduler']}'.")
        self._clear_previous_experiment()

        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.orchestrator.dispatch(ActionType.SET_CHALLENGE, {"name": dataset_name, "type": challenge_def.type})
        self.orchestrator.dispatch(ActionType.SET_ADAPTIVE_POLICY, {"policy_name": config["adaptive_scheduler"]})

        for model_name, model_params in config["models"].items():
            full_param_space = {}
            for param_type in model_params.values():
                for param_name, properties in param_type.items():
                    full_param_space[param_name] = {
                        "type": "float", "min": properties["min"], "max": properties["max"],
                        "scale": properties.get("scale", "linear"),
                    }
            self.orchestrator.dispatch(ActionType.ADD_ALGORITHM, {"name": model_name, "parameter_space": full_param_space})

        self.append_log_message(f"INFO: Starting run. The '{config['adaptive_scheduler']}' policy will now generate trials.")
        start_payload = {"enable_checkpointing": self.setup_pane.checkpoint_checkbox.isChecked()}
        self.orchestrator.dispatch(ActionType.START_RUN, start_payload)

    def _clear_previous_experiment(self):
        """Clears all UI elements and the ViewModel for a new experiment."""
        self.view_model.clear()
        self.trials_table.setRowCount(0)
        self.plot_widget.clear()
        self.trial_row_map.clear()
        self.plot_curve_map.clear()
        self.insights_list.clear()
        self.setup_pane.clear_algorithms_table()
        self.setup_plot()  # Re-add legend and titles

    def on_trial_selected(self):
        """Handles trial selection in the table, highlighting the corresponding plot."""
        selected_items = self.trials_table.selectedItems()
        if not selected_items:
            self._update_plot_highlight(set()) # Empty set clears all highlights
            return

        selected_row = self.trials_table.currentRow()
        selected_trial_id = None
        for tid, r in self.trial_row_map.items():
            if r == selected_row:
                selected_trial_id = tid
                break

        if selected_trial_id:
            self._update_plot_highlight({selected_trial_id})

    def update_trial_ui(self, ui_trial, metric_name: str):
        """Updates or creates a row in the trials table for a given UITrial."""
        trial_id = ui_trial.id
        if trial_id not in self.trial_row_map:
            row_position = self.trials_table.rowCount()
            self.trials_table.insertRow(row_position)
            self.trial_row_map[trial_id] = row_position

            # Create plot curve item
            name = f"{ui_trial.algorithm_name} ({trial_id[:6]})"
            pen = ui_trial.pen
            self.plot_curve_map[trial_id] = self.plot_widget.plot(
                [], [], name=name, pen=pen, symbol="o", symbolSize=6, symbolBrush=pen.color()
            )

        row = self.trial_row_map[trial_id]
        # Update plot pen and row color based on status (e.g., best, pruned)
        self.plot_curve_map[trial_id].setPen(ui_trial.pen)
        background_color = ui_trial.row_background_color

        # Populate table cells
        self.trials_table.setItem(row, 0, QTableWidgetItem(trial_id))
        self.trials_table.setItem(row, 1, QTableWidgetItem(ui_trial.algorithm_name))
        self.trials_table.setItem(row, 2, QTableWidgetItem(ui_trial.status))
        self.trials_table.setItem(row, 3, QTableWidgetItem(ui_trial.display_epoch))
        self.trials_table.setItem(row, 4, QTableWidgetItem(ui_trial.get_latest_metric(metric_name)))
        self.trials_table.setItem(row, 5, QTableWidgetItem(ui_trial.get_latest_metric("loss")))
        self.trials_table.setItem(row, 6, QTableWidgetItem(ui_trial.display_est_time))

        # Add a button to view hyperparameters
        view_button = QPushButton("View")
        view_button.clicked.connect(lambda: self.show_hyperparameter_dialog(ui_trial.hyperparameters))
        self.trials_table.setCellWidget(row, 7, view_button)

        for col in range(self.trials_table.columnCount()):
            if self.trials_table.item(row, col): # Only color items, not widgets
                self.trials_table.item(row, col).setBackground(background_color)

    def on_insight_selected(self, item: InsightListItem):
        """Highlights the trials relevant to the selected insight."""
        if not isinstance(item, InsightListItem):
            return

        if self.selected_insight_item == item:
            self.insights_list.clearSelection()
            self.selected_insight_item = None
            self._update_plot_highlight(set())
            return

        self.selected_insight_item = item
        highlight_ids = set(item.insight.trial_ids)
        self._update_plot_highlight(highlight_ids)

        # Also select the rows in the table
        self.trials_table.clearSelection()
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for trial_id, row in self.trial_row_map.items():
            if trial_id in highlight_ids:
                self.trials_table.selectRow(row)
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def _update_plot_highlight(self, highlight_ids: set):
        """Highlights a specific set of trials on the plot."""
        for trial_id, curve in self.plot_curve_map.items():
            ui_trial = self.view_model.trials.get(trial_id)
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

    def append_log_message(self, message: str):
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        """Handles the window close event to ensure graceful shutdown."""
        self.append_log_message("INFO: Close event received. Shutting down backend engine...")
        self.orchestrator.shutdown()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
