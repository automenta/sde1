import sys
import uuid
import json
import itertools
import random
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QSplitter,
    QPushButton, QSizePolicy, QComboBox, QLabel, QListWidget, QListWidgetItem,
    QFormLayout, QSlider, QCheckBox, QProgressBar, QGroupBox, QMessageBox, QDialog,
    QStyle
)
from PyQt6.QtGui import QIcon, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import pyqtgraph as pg

# Import backend components
from sde.engine.orchestrator import ExperimentOrchestrator
from sde.engine.insight import Insight
from sde.core.types import Trial
from sde.exploration.schedulers import SuccessiveHalvingScheduler, HyperbandScheduler
from sde.models import AVAILABLE_MODELS
from sde.challenges import AVAILABLE_DATASETS
from sde.ui.hyperparameters import HyperparameterDialog


from PyQt6.QtWidgets import QAbstractItemView


from datetime import datetime


from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QTreeWidget, QTreeWidgetItem


class InsightListItem(QListWidgetItem):
    """A custom QListWidgetItem that stores the full Insight object."""
    def __init__(self, insight: Insight, icon: QIcon, parent: QListWidget | None = None):
        super().__init__(parent)
        self.insight = insight
        self.setIcon(icon)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.setText(f"[{timestamp}] {self.insight.message}")
        # The full message is now displayed, but tooltip is still good for copy/paste.
        self.setToolTip(self.insight.message)


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
                    value_str = f"{value:.6g}" # Use general format for nice printing
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

        # --- V2 Backend components ---
        self.orchestrator = ExperimentOrchestrator()

        # --- Data maps for UI updates ---
        self.trial_row_map = {}  # trial.id -> table_row_index
        self.plot_curve_map = {} # trial.id -> plot_curve_item
        self.legend = None
        self.selected_insight_item = None
        self.best_trial_id = None
        self.current_state = {}
        self._setup_icons()

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Pane: Experiment Setup ---
        setup_pane = QWidget()
        setup_pane.setMaximumWidth(350)
        setup_layout = QVBoxLayout(setup_pane)

        # --- Right Pane: Results ---
        results_pane = QWidget()
        results_layout = QVBoxLayout(results_pane)

        main_layout.addWidget(setup_pane)
        main_layout.addWidget(results_pane, 1)

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
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Progress: %p%")

        self.start_button = QPushButton("Start (Defaults)")
        self.tune_button = QPushButton("Tune Hyperparameters...")
        self.add_models_button = QPushButton("Add Selected Models to Run")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")

        controls_group = QGroupBox("3. Execution Controls")
        controls_layout = QVBoxLayout(controls_group)
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

        # --- Assemble Left Pane ---
        setup_layout.addWidget(self.setup_group)
        setup_layout.addWidget(self.settings_group)
        setup_layout.addStretch(1)
        setup_layout.addWidget(self.progress_bar)
        setup_layout.addWidget(controls_group)

        # --- Connect Signals and Slots ---
        self.populate_datasets()
        self.dataset_combo.currentIndexChanged.connect(self.update_model_list)
        self.start_button.clicked.connect(self.start_simple_experiment)
        self.tune_button.clicked.connect(self.open_tuning_dialog)
        self.add_models_button.clicked.connect(self.add_models_to_run)
        self.pause_button.clicked.connect(self.pause_experiment)
        self.resume_button.clicked.connect(self.resume_experiment)
        self.throttle_slider.valueChanged.connect(self.update_throttle)

        # V2 Signal Connections
        self.orchestrator.log_message.connect(self.append_log_message)
        self.orchestrator.state_changed.connect(self.on_state_changed)

        self.update_model_list()
        self.update_button_states(running=False, paused=False)

        # --- Main Content Splitter (in results_pane)---
        splitter = QSplitter(Qt.Orientation.Vertical)
        results_layout.addWidget(splitter)
        self.plot_widget = pg.PlotWidget()
        self.setup_plot()

        bottom_pane = QWidget()
        bottom_layout = QHBoxLayout(bottom_pane)
        self.trials_table = QTableWidget()

        right_bottom_splitter = QSplitter(Qt.Orientation.Vertical)
        insights_group = QGroupBox("Insights")
        insights_layout = QVBoxLayout(insights_group)
        self.insights_list = QListWidget()
        self.insights_list.setWordWrap(True)  # Enable word wrapping for insight messages
        insights_layout.addWidget(self.insights_list)
        insights_layout.setContentsMargins(0, 5, 0, 0)
        insights_group.setLayout(insights_layout)
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
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self.trials_table)
        bottom_splitter.addWidget(right_bottom_splitter)
        bottom_splitter.setSizes([750, 450])
        bottom_layout.addWidget(bottom_splitter)
        bottom_pane.setLayout(bottom_layout)

        splitter.addWidget(self.plot_widget)
        splitter.addWidget(bottom_pane)
        splitter.setSizes([500, 300])

        self.setup_table()
        self.append_log_message("INFO: UI Initialized. Configure your experiment and click 'Start'.")

    def populate_datasets(self):
        self.dataset_combo.addItems(AVAILABLE_DATASETS.keys())

    def update_model_list(self):
        self.model_list.clear()
        selected_dataset_name = self.dataset_combo.currentText()
        if not selected_dataset_name: return
        dataset_def = AVAILABLE_DATASETS[selected_dataset_name]
        supported_models = [name for name, model_def in AVAILABLE_MODELS.items() if dataset_def.type in model_def.supported_dataset_types]
        for model_name in supported_models:
            item = QListWidgetItem(model_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.model_list.addItem(item)

    def setup_plot(self):
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Real-Time Trial Performance", color="k", size="16pt")
        self.plot_widget.setLabel('left', 'Accuracy', color='k', **{'font-size': '12pt'})
        self.plot_widget.setLabel('bottom', 'Epoch', color='k', **{'font-size': '12pt'})
        self.plot_widget.showGrid(x=True, y=True)
        self.legend = self.plot_widget.addLegend()

    def setup_table(self):
        self.trials_table.setColumnCount(8)
        self.trials_table.setHorizontalHeaderLabels(["Trial ID", "Algorithm", "Status", "Epoch", "Accuracy", "Loss", "Est. Time/Epoch", "Hyperparameters"])
        header = self.trials_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.trials_table.setColumnWidth(0, 100); self.trials_table.setColumnWidth(1, 120); self.trials_table.setColumnWidth(2, 100); self.trials_table.setColumnWidth(3, 60); self.trials_table.setColumnWidth(4, 100); self.trials_table.setColumnWidth(5, 100); self.trials_table.setColumnWidth(6, 120)
        self.trials_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.trials_table.itemSelectionChanged.connect(self.on_trial_selected)
        self.trials_table.cellDoubleClicked.connect(self.on_trial_double_clicked)
        self.insights_list.itemClicked.connect(self.on_insight_selected)

    def on_trial_double_clicked(self, row, column):
        """Shows the hyperparameter viewer for the double-clicked trial."""
        trial_id = None
        for tid, r in self.trial_row_map.items():
            if r == row:
                trial_id = tid
                break

        if trial_id:
            trials = self.current_state.get('trials', {})
            trial_data = trials.get(trial_id)
            if trial_data:
                hparams = trial_data.get('hyperparameters', {})
                dialog = HyperparameterViewerDialog(hparams, self)
                dialog.exec()

    def on_state_changed(self, state: dict):
        """
        The central handler for all state updates from the V2 Orchestrator.
        """
        self.current_state = state

        # V2 UI state management
        valid_actions = state.get('valid_actions', {})
        self.update_button_states(valid_actions)

        trials = state.get('trials', {})

        # Update table and plots
        for trial_data in trials.values():
            self.update_trial_ui(trial_data) # This method now primarily handles the table

        self.update_plots(trials) # This new method handles updating the plots

        # Update progress bar
        if trials:
            total_trials = len(trials)
            completed_statuses = {"COMPLETED", "PRUNED"}
            completed_trials = sum(1 for t in trials.values() if t['status'] in completed_statuses)
            progress = int((completed_trials / total_trials) * 100)
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.setValue(0)

        # --- Find and store best trial ---
        dataset_name = self.dataset_combo.currentText()
        if dataset_name:
            challenge_def = AVAILABLE_DATASETS[dataset_name]
            metric_name = challenge_def.performance_metric_name
            higher_is_better = 'accuracy' in metric_name.lower() # Simple inference

            best_trial_id = None
            best_perf = -float('inf') if higher_is_better else float('inf')

            for trial_id, trial_data in trials.items():
                if trial_data.get('results', {}).get(metric_name):
                    latest_perf = trial_data['results'][metric_name][-1][1]
                    if (higher_is_better and latest_perf > best_perf) or \
                       (not higher_is_better and latest_perf < best_perf):
                        best_perf = latest_perf
                        best_trial_id = trial_id

            self.best_trial_id = best_trial_id

        # Update insights
        insights = state.get('insights', [])
        if not hasattr(self, 'displayed_insight_messages'):
            self.displayed_insight_messages = set()

        for insight_data in insights:
            # Use message as a unique key to avoid displaying duplicates
            if insight_data['message'] not in self.displayed_insight_messages:
                # The add_insight method expects an Insight object, not a dict.
                # Re-create the object from the dictionary.
                insight_obj = Insight(
                    message=insight_data['message'],
                    type=insight_data['type'],
                    trial_ids=insight_data['trial_ids']
                )
                self.add_insight(insight_obj)
                self.displayed_insight_messages.add(insight_obj.message)


    def update_plots(self, trials_data: dict):
        """Updates all plot curves based on the latest trial data."""
        dataset_name = self.dataset_combo.currentText()
        if not dataset_name:
            return

        metric_name = AVAILABLE_DATASETS[dataset_name].performance_metric_name

        for trial_id, trial_data in trials_data.items():
            if trial_id in self.plot_curve_map:
                metric_list = trial_data.get('results', {}).get(metric_name, [])
                if metric_list:
                    # Ensure data is in a format that can be plotted
                    try:
                        epochs, metrics = zip(*metric_list)
                        self.plot_curve_map[trial_id].setData(epochs, metrics)
                    except ValueError:
                        # Handle cases with empty or malformed metric_list
                        self.plot_curve_map[trial_id].clear()


    def update_button_states(self, valid_actions: dict):
        """
        Updates the enabled/disabled state of UI controls based on the
        valid actions provided by the orchestrator. This is the core of the
        V2 modeless UI.
        """
        global_actions = valid_actions.get('global', [])
        status = self.current_state.get('status')

        # --- Setup Controls ---
        # These controls are used to define the experiment before it starts.
        # They are enabled based on the actions available in the DEFINING state.
        self.dataset_combo.setEnabled("SET_CHALLENGE" in global_actions)
        self.model_list.setEnabled("ADD_ALGORITHM" in global_actions)
        self.start_button.setEnabled("START_RUN" in global_actions)
        self.tune_button.setEnabled("START_RUN" in global_actions)

        # The settings group doesn't map to a specific action, but is only
        # relevant before a run starts.
        self.settings_group.setEnabled(status == "DEFINING")

        # --- Mid-Run Controls ---
        # These controls are for interacting with a live experiment.
        self.pause_button.setEnabled("PAUSE_RUN" in global_actions)
        self.resume_button.setEnabled("RESUME_RUN" in global_actions)

        # The "Add Models" button is specifically for mid-run additions.
        # The ADD_ALGORITHM action is also valid in the DEFINING state, but in that
        # context, model selection is handled by the main list and start buttons.
        # Therefore, we also check the status here to correctly map the button's intent.
        can_add_mid_run = "ADD_ALGORITHM" in global_actions and status in ["RUNNING", "PAUSED"]
        self.add_models_button.setEnabled(can_add_mid_run)


    def update_throttle(self, value: int):
        self.throttle_label.setText(f"{value}%")
        # Throttling is not currently connected in the V2 orchestrator.
        # This could be a future feature implemented via a SET_THROTTLE action.

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
            self.append_log_message("ERROR: Please select a dataset and at least one model.")
            return

        self.append_log_message(f"INFO: Configuring experiment on '{dataset_name}' with models: {selected_models}")
        self._clear_previous_experiment()

        # --- Dispatch Actions ---
        # 1. Set Challenge
        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.orchestrator.dispatch("SET_CHALLENGE", {"name": dataset_name, "type": challenge_def.type})

        # 2. Add Algorithms
        for model_name in selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            # Create a simplified parameter space for the simple experiment
            param_space = {
                k: (v['min'], v['max']) for param_type in model_def.hyperparameter_schema.values()
                for k, v in param_type.items()
            }
            self.orchestrator.dispatch("ADD_ALGORITHM", {"name": model_name, "parameter_space": param_space})

        # 3. Start the Run
        self.orchestrator.dispatch("START_RUN", {})

    def add_models_to_run(self):
        """
        Adds newly selected models to an already running experiment.
        """
        # Get all currently checked models
        all_selected_models = {self.model_list.item(i).text() for i in range(self.model_list.count()) if self.model_list.item(i).checkState() == Qt.CheckState.Checked}

        # Get models that are already part of the experiment
        if not self.current_state:
            self.append_log_message("ERROR: No current state available to add models to.")
            return
        existing_algo_names = {algo['name'] for algo in self.current_state.get('algorithms', {}).values()}

        # Determine which models are new
        newly_selected_models = all_selected_models - existing_algo_names

        if not newly_selected_models:
            self.append_log_message("INFO: No new models selected to add.")
            return

        self.append_log_message(f"INFO: Adding new models to run: {', '.join(newly_selected_models)}")

        for model_name in newly_selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = {
                k: (v['min'], v['max']) for param_type in model_def.hyperparameter_schema.values()
                for k, v in param_type.items()
            }
            self.orchestrator.dispatch("ADD_ALGORITHM", {"name": model_name, "parameter_space": param_space})


    def open_tuning_dialog(self):
        dataset_name, selected_models_names = self._get_experiment_settings()
        if not dataset_name or not selected_models_names:
            QMessageBox.warning(self, "Warning", "Please select a dataset and at least one model to tune.")
            return

        selected_model_defs = [AVAILABLE_MODELS[name] for name in selected_models_names]
        dialog = HyperparameterDialog(selected_model_defs, self)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_configuration()
        self.append_log_message(f"INFO: Configuring tuning experiment with scheduler '{config['adaptive_scheduler']}' and h-param strategy '{config['hparam_strategy']}'")
        self._clear_previous_experiment()

        # --- V2 Dispatch Logic ---
        # 1. Set Challenge
        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.orchestrator.dispatch("SET_CHALLENGE", {"name": dataset_name, "type": challenge_def.type})

        # 2. Generate parameter combinations and dispatch ADD_ALGORITHM for each
        param_combinations = self._generate_param_combinations(config)

        if not param_combinations:
            self.append_log_message("ERROR: Tuning configuration did not generate any parameter combinations.")
            return

        for model_name, hparams in param_combinations:
             # The "parameter space" for this action is just the single, fixed point
             # from the hyperparameter search.
            self.orchestrator.dispatch("ADD_ALGORITHM", {
                "name": model_name,
                "parameter_space": hparams
            })

        # 3. Start the run
        self.append_log_message(f"INFO: Starting run with {len(param_combinations)} trial configurations.")
        self.orchestrator.dispatch("START_RUN", {})

    def _generate_param_combinations(self, config: dict) -> list[tuple[str, dict]]:
        """
        Generates a list of (model_name, hparams) tuples based on the
        tuning configuration from the dialog.
        """
        strategy = config.get('hparam_strategy', 'Grid Search')
        num_trials = config.get('num_trials', 1)
        all_combinations = []

        for model_name, model_config in config.get('models', {}).items():
            hparam_def = {}
            # Flatten the param definition for easier processing
            for param_type, params in model_config.items():
                for param_name, properties in params.items():
                    hparam_def[param_name] = properties

            if not hparam_def:
                continue

            if strategy == 'Grid Search':
                # Create a list of value lists for grid search
                param_grid = {
                    k: np.linspace(v['min'], v['max'], num_trials) if v.get('scale') != 'log'
                    else np.logspace(np.log10(v['min']), np.log10(v['max']), num_trials)
                    for k, v in hparam_def.items()
                }
                keys, values = zip(*param_grid.items())
                combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
                for combo in combinations:
                    all_combinations.append((model_name, combo))
            else:  # Random Search
                for _ in range(num_trials):
                    combination = {}
                    for param_name, properties in hparam_def.items():
                        if properties.get('scale') == 'log':
                            log_min = np.log10(properties['min'])
                            log_max = np.log10(properties['max'])
                            value = 10**random.uniform(log_min, log_max)
                        else:
                            value = random.uniform(properties['min'], properties['max'])
                        combination[param_name] = value
                    all_combinations.append((model_name, combination))

        return all_combinations
    def _get_experiment_settings(self):
        dataset_name = self.dataset_combo.currentText()
        selected_models = [self.model_list.item(i).text() for i in range(self.model_list.count()) if self.model_list.item(i).checkState() == Qt.CheckState.Checked]
        return dataset_name, selected_models

    def _clear_previous_experiment(self):
        self.trials_table.setRowCount(0)
        self.plot_widget.clear()
        self.trial_row_map.clear()
        self.plot_curve_map.clear()
        self.insights_list.clear()
        self.setup_plot()

    def on_trial_selected(self):
        selected_items = self.trials_table.selectedItems()
        if not selected_items:
            for curve in self.plot_curve_map.values():
                original_pen = curve.opts['pen']
                original_color = original_pen.color()
                original_color.setAlpha(255)
                curve.setPen(pg.mkPen(color=original_color, width=2))
                curve.setZValue(0)
            return
        selected_row = self.trials_table.currentRow()
        selected_trial_id = None
        for tid, r in self.trial_row_map.items():
            if r == selected_row:
                selected_trial_id = tid
                break
        if selected_trial_id:
            self._update_plot_highlight({selected_trial_id})

    def on_trial_profiled(self, trial_id: str, est_time: float):
        if trial_id in self.trial_row_map:
            row = self.trial_row_map[trial_id]
            self.trials_table.setItem(row, 6, QTableWidgetItem(f"{est_time:.2f}s"))

    def update_trial_ui(self, trial_data: dict):
        trial_id = trial_data['id']
        if trial_id not in self.trial_row_map:
            row_position = self.trials_table.rowCount()
            self.trials_table.insertRow(row_position)
            self.trial_row_map[trial_id] = row_position
            color = pg.intColor(len(self.plot_curve_map), hues=9, values=1)
            name = f"{trial_data['algorithm_name']} ({trial_id})"
            if self.legend and self.legend.isVisible():
                curve_name = name
            else:
                curve_name = None
            pen = pg.mkPen(color=color, width=2)
            self.plot_curve_map[trial_id] = self.plot_widget.plot([], [], name=curve_name, pen=pen, symbol='o', symbolSize=6, symbolBrush=pen.color())

        row = self.trial_row_map[trial_id]
        self.trials_table.setItem(row, 0, QTableWidgetItem(trial_id))
        self.trials_table.setItem(row, 1, QTableWidgetItem(trial_data['algorithm_name']))
        status = trial_data['status']
        self.trials_table.setItem(row, 2, QTableWidgetItem(status))
        self.trials_table.setItem(row, 3, QTableWidgetItem(str(trial_data['current_epoch'])))
        dataset_name = self.dataset_combo.currentText()
        metric_name = AVAILABLE_DATASETS[dataset_name].performance_metric_name
        metric_list = trial_data['results'].get(metric_name, [])
        loss_list = trial_data['results'].get('loss', [])
        latest_metric = f"{metric_list[-1][1]:.4f}" if metric_list else "N/A"
        latest_loss = f"{loss_list[-1][1]:.4f}" if loss_list else "N/A"
        self.trials_table.setItem(row, 4, QTableWidgetItem(latest_metric))
        self.trials_table.setItem(row, 5, QTableWidgetItem(latest_loss))

        est_time = trial_data.get('est_time_per_epoch')
        if est_time is not None:
             self.trials_table.setItem(row, 6, QTableWidgetItem(f"{est_time:.2f}s"))

        self.trials_table.setItem(row, 7, QTableWidgetItem(json.dumps(trial_data['hyperparameters'])))

        self._style_trial_ui(trial_id, status)

    def _setup_icons(self):
        """Pre-loads icons for different insight types."""
        style = self.style()
        self.insight_icons = {
            "BEST_PERFORMER": style.standardIcon(QStyle.StandardPixmap.SP_ArrowUp),
            "PLATEAU": style.standardIcon(QStyle.StandardPixmap.SP_ArrowRight),
            "POOR_INITIAL_PERFORMANCE": style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning),
            "PERFORMANCE_CROSSOVER": style.standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward),
            "HYPERPARAM_CORRELATION": style.standardIcon(QStyle.StandardPixmap.SP_DialogHelpButton),
            "DEFAULT": style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
        }

    def on_insight_selected(self, item: InsightListItem):
        """Highlights the trials relevant to the selected insight."""
        if not isinstance(item, InsightListItem):
            return

        if self.selected_insight_item == item:
            self.insights_list.clearSelection()
            self.selected_insight_item = None
            self.on_trial_selected()
            return

        self.selected_insight_item = item
        highlight_ids = set(item.insight.trial_ids)
        if not highlight_ids:
            return

        self.trials_table.clearSelection()
        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

        for trial_id, row in self.trial_row_map.items():
            if trial_id in highlight_ids:
                self.trials_table.selectRow(row)

        self.trials_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._update_plot_highlight(highlight_ids)

    def _update_plot_highlight(self, highlight_ids: set):
        """Highlights a specific set of trials on the plot."""
        for trial_id, curve in self.plot_curve_map.items():
            pen = curve.opts['pen']
            color = pen.color()
            if trial_id in highlight_ids:
                color.setAlpha(255)
                curve.setPen(pg.mkPen(color=color, width=4))
                curve.setZValue(100)
            else:
                color.setAlpha(30)
                curve.setPen(pg.mkPen(color=color, width=1))
                curve.setZValue(0)

    def _style_trial_ui(self, trial_id: str, status: str):
        """Applies coloring and styling to a trial's row and plot based on its status."""
        row_color = QColor('white')
        pen = self.plot_curve_map[trial_id].opts['pen']

        is_best = (self.best_trial_id == trial_id)

        if is_best:
            row_color = QColor('#FFFACD')  # LemonChiffon
            pen.setColor(pg.mkColor('#FFD700')) # Gold
            pen.setWidth(4)
            pen.setStyle(Qt.PenStyle.SolidLine)
        elif status == "PRUNED":
            row_color = QColor('#D3D3D3')
            pen.setColor(pg.mkColor('#808080'))
            pen.setStyle(Qt.PenStyle.DotLine)
        elif status == "COMPLETED":
            row_color = QColor('#ADD8E6')
            pen.setColor(pg.mkColor('#0000FF'))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.SolidLine)
        else: # ACTIVE
            original_color = pg.intColor(list(self.plot_curve_map.keys()).index(trial_id), hues=9, values=1)
            pen.setColor(original_color)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.SolidLine)

        self.plot_curve_map[trial_id].setPen(pen)

        row = self.trial_row_map[trial_id]
        for col in range(self.trials_table.columnCount()):
            self.trials_table.item(row, col).setBackground(row_color)

    def add_insight(self, insight: Insight):
        icon = self.insight_icons.get(insight.type, self.insight_icons["DEFAULT"])
        item = InsightListItem(insight, icon, self.insights_list)
        self.insights_list.addItem(item)
        self.insights_list.scrollToBottom()

    def append_log_message(self, message: str):
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def closeEvent(self, event):
        # V2: The orchestrator handles the lifecycle of the runtime engine.
        # We might need a "shutdown" or "cleanup" action in the future,
        # but for now, letting the process exit is sufficient as the
        # engine thread is a daemon.
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
