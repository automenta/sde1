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
from sde.engine.orchestrator import ExperimentOrchestrator
from sde.models import AVAILABLE_MODELS
from sde.challenges import AVAILABLE_DATASETS
from sde.ui.hyperparameters import HyperparameterDialog
from sde.ui.view_model import ExperimentViewModel
from sde.ui.models import UIInsight


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
        self.populate_datasets()
        self.update_model_list()
        self.update_button_states() # Initialize button states
        self.append_log_message(
            "INFO: UI Initialized. Configure your experiment and click 'Start'."
        )

    def _init_ui(self):
        """Initializes the main UI layout and sub-components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_pane = self._create_left_pane()
        right_pane = self._create_right_pane()

        main_layout.addWidget(left_pane)
        main_layout.addWidget(right_pane, 1)

    def _create_left_pane(self) -> QWidget:
        """Creates the left-hand pane for experiment setup and controls."""
        setup_pane = QWidget()
        setup_pane.setMaximumWidth(350)
        setup_layout = QVBoxLayout(setup_pane)

        # --- Experiment Setup Group ---
        self.setup_group = QGroupBox("1. Experiment Setup")
        setup_form_layout = QFormLayout(self.setup_group)
        self.dataset_combo = QComboBox()
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.model_list.setMinimumHeight(150)
        setup_form_layout.addRow("Dataset:", self.dataset_combo)
        setup_form_layout.addRow("Models:", self.model_list)

        # --- Execution Settings Group ---
        self.settings_group = QGroupBox("2. Execution Settings")
        settings_form_layout = QFormLayout(self.settings_group)
        self.throttle_slider = QSlider(Qt.Orientation.Horizontal)
        self.throttle_slider.setRange(1, 100)
        self.throttle_slider.setValue(100)
        self.throttle_label = QLabel("100%")
        throttle_widget = QWidget()
        throttle_layout = QHBoxLayout(throttle_widget)
        throttle_layout.addWidget(self.throttle_slider)
        throttle_layout.addWidget(self.throttle_label)
        throttle_layout.setContentsMargins(0, 0, 0, 0)
        self.checkpoint_checkbox = QCheckBox("Enable Checkpointing")
        self.checkpoint_checkbox.setChecked(False)
        settings_form_layout.addRow("Worker Throttle:", throttle_widget)
        settings_form_layout.addRow(self.checkpoint_checkbox)

        # --- Execution Controls ---
        controls_group = QGroupBox("3. Execution Controls")
        controls_layout = QVBoxLayout(controls_group)
        self.start_button = QPushButton("Start (Defaults)")
        self.start_button.setToolTip("Run the selected models with their default hyperparameter ranges.")
        self.tune_button = QPushButton("Tune Hyperparameters...")
        self.tune_button.setToolTip("Define custom hyperparameter spaces and select an advanced tuning policy.")
        self.add_models_button = QPushButton("Add Selected Models to Run")
        self.add_models_button.setToolTip("Add newly selected models to the currently running experiment.")
        self.pause_button = QPushButton("Pause")
        self.pause_button.setToolTip("Pause the currently running experiment.")
        self.resume_button = QPushButton("Resume")
        self.resume_button.setToolTip("Resume a paused experiment.")
        simple_run_layout = QHBoxLayout()
        simple_run_layout.addWidget(self.start_button)
        advanced_run_layout = QHBoxLayout()
        advanced_run_layout.addWidget(self.tune_button)
        mid_run_layout = QHBoxLayout()
        mid_run_layout.addWidget(self.add_models_button)
        pause_resume_layout = QHBoxLayout()
        pause_resume_layout.addWidget(self.pause_button)
        pause_resume_layout.addWidget(self.resume_button)
        controls_layout.addLayout(simple_run_layout)
        controls_layout.addLayout(advanced_run_layout)
        controls_layout.addLayout(mid_run_layout)
        controls_layout.addLayout(pause_resume_layout)

        # --- Algorithm Management Group ---
        self.algorithms_group = QGroupBox("4. Active Algorithms")
        algorithms_layout = QVBoxLayout(self.algorithms_group)
        self.algorithms_table = QTableWidget()
        self.algorithms_table.setColumnCount(2)
        self.algorithms_table.setHorizontalHeaderLabels(["Name", "Actions"])
        header = self.algorithms_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        algorithms_layout.addWidget(self.algorithms_table)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Progress: %p%")

        # --- Assemble Left Pane ---
        setup_layout.addWidget(self.setup_group)
        setup_layout.addWidget(self.settings_group)
        setup_layout.addWidget(controls_group)
        setup_layout.addWidget(self.algorithms_group)
        setup_layout.addStretch(1)
        setup_layout.addWidget(self.progress_bar)

        return setup_pane

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

        # Left pane controls
        self.dataset_combo.currentIndexChanged.connect(self.update_model_list)
        self.start_button.clicked.connect(self.start_simple_experiment)
        self.tune_button.clicked.connect(self.open_tuning_dialog)
        self.add_models_button.clicked.connect(self.add_models_to_run)
        self.pause_button.clicked.connect(self.pause_experiment)
        self.resume_button.clicked.connect(self.resume_experiment)
        self.throttle_slider.valueChanged.connect(self.update_throttle)

        # Right pane controls
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

    def populate_datasets(self):
        self.dataset_combo.addItems(AVAILABLE_DATASETS.keys())

    def update_model_list(self):
        self.model_list.clear()
        selected_dataset_name = self.dataset_combo.currentText()
        if not selected_dataset_name:
            return
        dataset_def = AVAILABLE_DATASETS[selected_dataset_name]
        supported_models = [
            name
            for name, model_def in AVAILABLE_MODELS.items()
            if dataset_def.type in model_def.supported_dataset_types
        ]
        for model_name in supported_models:
            item = QListWidgetItem(model_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.model_list.addItem(item)

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
        self.update_button_states()
        self.update_algorithm_table()
        self.update_trials_and_plots()
        self.update_progress_bar()
        self.update_insights_list()

    def update_algorithm_table(self):
        """Updates the algorithm table from the ViewModel."""
        self.algorithms_table.setRowCount(0)
        algo_actions = self.view_model.valid_actions.get("algorithms", {})

        for algo_id, algo_data in self.view_model.algorithms.items():
            row_position = self.algorithms_table.rowCount()
            self.algorithms_table.insertRow(row_position)
            self.algorithms_table.setItem(row_position, 0, QTableWidgetItem(algo_data.name))

            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            remove_button = QPushButton("Remove")
            remove_button.setEnabled("REMOVE_ALGORITHM" in algo_actions.get(algo_id, []))
            remove_button.clicked.connect(
                lambda _, a_id=algo_id: self.orchestrator.dispatch(
                    "REMOVE_ALGORITHM", {"algorithm_id": a_id}
                )
            )
            actions_layout.addWidget(remove_button)
            self.algorithms_table.setCellWidget(row_position, 1, actions_widget)

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
            self.progress_bar.setValue(0)
            return

        total_trials = len(trials)
        completed_statuses = {"COMPLETED", "PRUNED"}
        completed_trials = sum(1 for t in trials if t.status in completed_statuses)
        progress = int((completed_trials / total_trials) * 100)
        self.progress_bar.setValue(progress)

    def update_insights_list(self):
        """Updates the insights list from the ViewModel."""
        # A simple approach: clear and redraw. More complex logic could be used
        # to avoid flickering, but this is robust for now.
        self.insights_list.clear()
        for ui_insight in self.view_model.insights:
            item = InsightListItem(ui_insight, self.insights_list)
            self.insights_list.addItem(item)
        self.insights_list.scrollToBottom()

    def update_button_states(self):
        """Updates button states based on the ViewModel's `valid_actions`."""
        valid_actions = self.view_model.valid_actions
        global_actions = valid_actions.get("global", [])
        status = self.view_model.status
        has_challenge = bool(self.view_model.challenge_name)

        can_start = "START_RUN" in global_actions
        self.start_button.setEnabled(can_start)
        self.tune_button.setEnabled(can_start)

        self.pause_button.setEnabled("PAUSE_RUN" in global_actions)
        self.resume_button.setEnabled("RESUME_RUN" in global_actions)

        can_add_mid_run = "ADD_ALGORITHM" in global_actions and status in ["RUNNING", "PAUSED"]
        self.add_models_button.setEnabled(can_add_mid_run)

        self.dataset_combo.setEnabled(not has_challenge)
        self.model_list.setEnabled("ADD_ALGORITHM" in global_actions)
        self.settings_group.setEnabled(status == "DEFINING")

    def update_throttle(self, value: int):
        self.throttle_label.setText(f"{value}%")
        # Dispatch a SET_BUDGET action. This is a conceptual mapping for now.
        # A more detailed implementation might have a richer budget definition.
        self.orchestrator.dispatch("SET_BUDGET", {"worker_throttle_percent": value})

    def pause_experiment(self):
        self.orchestrator.dispatch("PAUSE_RUN", {})

    def resume_experiment(self):
        self.orchestrator.dispatch("RESUME_RUN", {})

    def start_simple_experiment(self):
        """
        V2 implementation: Dispatches actions to the orchestrator to build
        and start an experiment.
        """
        dataset_name, selected_models = self._get_experiment_settings()
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
        # 1. Set Challenge
        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.orchestrator.dispatch(
            "SET_CHALLENGE", {"name": dataset_name, "type": challenge_def.type}
        )

        # 2. Add Algorithms
        for model_name in selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            # Create a simplified parameter space for the simple experiment
            param_space = {
                k: (v["min"], v["max"])
                for param_type in model_def.hyperparameter_schema.values()
                for k, v in param_type.items()
            }
            self.orchestrator.dispatch(
                "ADD_ALGORITHM", {"name": model_name, "parameter_space": param_space}
            )

        # 3. Start the Run, including execution settings
        start_payload = {"enable_checkpointing": self.checkpoint_checkbox.isChecked()}
        self.orchestrator.dispatch("START_RUN", start_payload)

    def add_models_to_run(self):
        """Adds newly selected models to an already running experiment."""
        all_selected_models = {
            self.model_list.item(i).text()
            for i in range(self.model_list.count())
            if self.model_list.item(i).checkState() == Qt.CheckState.Checked
        }

        # Get existing algorithm names from the ViewModel
        existing_algo_names = {algo.name for algo in self.view_model.algorithms.values()}
        newly_selected_models = all_selected_models - existing_algo_names

        if not newly_selected_models:
            self.append_log_message("INFO: No new models selected to add.")
            return

        self.append_log_message(f"INFO: Adding new models to run: {', '.join(newly_selected_models)}")
        for model_name in newly_selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = {
                k: (v["min"], v["max"])
                for param_type in model_def.hyperparameter_schema.values()
                for k, v in param_type.items()
            }
            self.orchestrator.dispatch("ADD_ALGORITHM", {"name": model_name, "parameter_space": param_space})

    def open_tuning_dialog(self):
        """Opens the tuning dialog and configures the experiment via the orchestrator."""
        dataset_name, selected_models_names = self._get_experiment_settings()
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
        self.orchestrator.dispatch("SET_CHALLENGE", {"name": dataset_name, "type": challenge_def.type})
        self.orchestrator.dispatch("SET_ADAPTIVE_POLICY", {"policy_name": config["adaptive_scheduler"]})

        for model_name, model_params in config["models"].items():
            full_param_space = {}
            for param_type in model_params.values():
                for param_name, properties in param_type.items():
                    full_param_space[param_name] = {
                        "type": "float", "min": properties["min"], "max": properties["max"],
                        "scale": properties.get("scale", "linear"),
                    }
            self.orchestrator.dispatch("ADD_ALGORITHM", {"name": model_name, "parameter_space": full_param_space})

        self.append_log_message(f"INFO: Starting run. The '{config['adaptive_scheduler']}' policy will now generate trials.")
        start_payload = {"enable_checkpointing": self.checkpoint_checkbox.isChecked()}
        self.orchestrator.dispatch("START_RUN", start_payload)

    def _get_experiment_settings(self):
        dataset_name = self.dataset_combo.currentText()
        selected_models = [
            self.model_list.item(i).text()
            for i in range(self.model_list.count())
            if self.model_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        return dataset_name, selected_models

    def _clear_previous_experiment(self):
        """Clears all UI elements and the ViewModel for a new experiment."""
        self.view_model.clear()
        self.trials_table.setRowCount(0)
        self.plot_widget.clear()
        self.trial_row_map.clear()
        self.plot_curve_map.clear()
        self.insights_list.clear()
        self.algorithms_table.setRowCount(0)
        self.setup_plot() # Re-add legend and titles

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
        # The SdeRuntimeEngine runs in a daemon thread, so it will be
        # automatically terminated when the main application exits.
        # A more advanced implementation could add a shutdown hook here
        # to tell the orchestrator to gracefully stop the engine.
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
