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
    QFormLayout, QSlider, QCheckBox, QProgressBar, QGroupBox, QMessageBox, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import pyqtgraph as pg

# Import backend components
from sde.engine.scheduler import Scheduler
from sde.core.types import Trial
from sde.exploration.schedulers import SuccessiveHalvingScheduler
from sde.models import AVAILABLE_MODELS
from sde.challenges import AVAILABLE_DATASETS
from sde.ui.hyperparameters import HyperparameterDialog

class SchedulerRunner(QObject):
    """
    A QObject wrapper that runs the framework-agnostic Scheduler in a QThread
    and translates its custom signals into PyQt signals.
    """
    # PyQt signals that will be emitted from the UI thread
    log_message = pyqtSignal(str)
    trial_updated = pyqtSignal(dict)
    trial_profiled = pyqtSignal(str, float)
    experiment_finished = pyqtSignal()
    insight_generated = pyqtSignal(str)

    def __init__(self, scheduler: Scheduler):
        super().__init__()
        self.scheduler = scheduler
        self._is_running = True

        # Connect the scheduler's custom signals to methods that emit PyQt signals
        self.scheduler.log_message.connect(self.log_message.emit)
        self.scheduler.trial_updated.connect(self.trial_updated.emit)
        self.scheduler.trial_profiled.connect(self.trial_profiled.emit)
        self.scheduler.experiment_finished.connect(self.on_scheduler_finished)
        self.scheduler.insight_generated.connect(self.insight_generated.emit)

    def run(self):
        """The main work method that is executed in the QThread."""
        if self._is_running:
            self.scheduler.start()

    def stop(self):
        """Stops the scheduler gracefully."""
        self.log_message.emit("INFO: UI requested scheduler shutdown.")
        self._is_running = False
        self.scheduler.stop()

    def on_scheduler_finished(self):
        """Handles the scheduler's finished signal."""
        self._is_running = False
        self.experiment_finished.emit()


from PyQt6.QtWidgets import QAbstractItemView


from datetime import datetime


class MainWindow(QMainWindow):
    """
    The main window for the Scientific Discovery Engine UI.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Discovery Engine")
        self.setGeometry(100, 100, 1400, 900)

        # --- Backend components ---
        self.scheduler_thread = None
        self.scheduler_runner = None

        # --- Data maps for UI updates ---
        self.trial_row_map = {}  # trial.id -> table_row_index
        self.plot_curve_map = {} # trial.id -> plot_curve_item
        self.legend = None

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
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")

        controls_group = QGroupBox("3. Execution Controls")
        controls_layout = QVBoxLayout(controls_group)
        simple_run_layout = QHBoxLayout()
        simple_run_layout.addWidget(self.start_button)
        advanced_run_layout = QHBoxLayout()
        advanced_run_layout.addWidget(self.tune_button)
        pause_resume_layout = QHBoxLayout()
        pause_resume_layout.addWidget(self.pause_button)
        pause_resume_layout.addWidget(self.resume_button)
        controls_layout.addLayout(simple_run_layout)
        controls_layout.addLayout(advanced_run_layout)
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
        self.pause_button.clicked.connect(self.pause_experiment)
        self.resume_button.clicked.connect(self.resume_experiment)
        self.throttle_slider.valueChanged.connect(self.update_throttle)

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

    def update_button_states(self, running: bool, paused: bool):
        self.start_button.setEnabled(not running)
        self.tune_button.setEnabled(not running)
        self.pause_button.setEnabled(running and not paused)
        self.resume_button.setEnabled(running and paused)
        self.setup_group.setEnabled(not running)
        self.settings_group.setEnabled(not running)

    def update_throttle(self, value: int):
        self.throttle_label.setText(f"{value}%")
        if self.scheduler_runner:
            self.scheduler_runner.scheduler.set_throttle(value)

    def pause_experiment(self):
        if self.scheduler_runner:
            self.scheduler_runner.scheduler.pause()
            self.update_button_states(running=True, paused=True)

    def resume_experiment(self):
        if self.scheduler_runner:
            self.scheduler_runner.scheduler.resume()
            self.update_button_states(running=True, paused=False)

    def start_simple_experiment(self):
        dataset_name, selected_models = self._get_experiment_settings()
        if not dataset_name or not selected_models:
            self.append_log_message("ERROR: Please select a dataset and at least one model.")
            return

        self.update_button_states(running=True, paused=False)
        self.append_log_message(f"INFO: Starting simple experiment on '{dataset_name}' with models: {selected_models}")
        self.progress_bar.setValue(0)
        self._clear_previous_experiment()
        trials = self._create_default_trials(selected_models)
        self._setup_and_run_scheduler(trials, dataset_name)

    def open_tuning_dialog(self):
        dataset_name, selected_models_names = self._get_experiment_settings()
        if not dataset_name or not selected_models_names:
            QMessageBox.warning(self, "Warning", "Please select a dataset and at least one model to tune.")
            return

        selected_model_defs = [AVAILABLE_MODELS[name] for name in selected_models_names]
        dialog = HyperparameterDialog(selected_model_defs, self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_configuration()
            self.append_log_message(f"INFO: Starting hyperparameter tuning with strategy: {config['strategy']}")

            template_trials = self._create_default_trials(selected_models_names)
            tuned_trials = self._generate_trials_from_config(template_trials, config)

            if not tuned_trials:
                self.append_log_message("ERROR: Tuning configuration did not generate any trials.")
                return

            if len(tuned_trials) > 15:
                self.legend.setVisible(False)
            else:
                self.legend.setVisible(True)

            self.update_button_states(running=True, paused=False)
            self._clear_previous_experiment()
            self._setup_and_run_scheduler(tuned_trials, dataset_name)

    def _generate_trials_from_config(self, template_trials: list, config: dict) -> list:
        strategy = config.get('strategy', 'Grid Search')
        steps = config.get('num_trials', 1)
        all_new_trials = []
        for template_trial in template_trials:
            model_config = config.get('models', {}).get(template_trial.algorithm_name)
            if not model_config:
                all_new_trials.append(template_trial)
                continue
            hparam_space = {}
            for param_type, params in model_config.items():
                for param_name, properties in params.items():
                    space_key = (param_type, param_name)
                    if properties['scale'] == 'log':
                        space = np.logspace(np.log10(properties['min']), np.log10(properties['max']), steps)
                    else:
                        space = np.linspace(properties['min'], properties['max'], steps)
                    hparam_space[space_key] = space
            if not hparam_space:
                all_new_trials.append(template_trial)
                continue
            keys, values = zip(*hparam_space.items())
            if strategy == 'Grid Search':
                param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            else:
                total_random_trials = config.get('num_trials', 1)
                param_combinations = []
                model_hparams_config = config.get('models', {}).get(template_trial.algorithm_name, {})
                for _ in range(total_random_trials):
                    combination = {}
                    for param_type, params in model_hparams_config.items():
                        for param_name, properties in params.items():
                            min_val = properties['min']
                            max_val = properties['max']
                            scale = properties['scale']
                            if scale == 'log':
                                log_min = np.log10(min_val)
                                log_max = np.log10(max_val)
                                value = 10**random.uniform(log_min, log_max)
                            else:
                                value = random.uniform(min_val, max_val)
                            combination[(param_type, param_name)] = value
                    param_combinations.append(combination)
            for combination in param_combinations:
                new_hparams = {k: v.copy() for k, v in template_trial.hyperparameters.items()}
                for (param_type, param_name), value in combination.items():
                    if param_type not in new_hparams: new_hparams[param_type] = {}
                    new_hparams[param_type][param_name] = value
                all_new_trials.append(Trial(id=f"{template_trial.algorithm_name[:4]}_t_{uuid.uuid4().hex[:4]}", algorithm_name=template_trial.algorithm_name, hyperparameters=new_hparams))
        return all_new_trials

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

    def _create_default_trials(self, selected_models: list) -> list:
        trials = []
        for model_name in selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            default_hparams = {}
            if "model_params" in model_def.hyperparameter_schema:
                default_hparams['model_params'] = {k: v['default'] for k, v in model_def.hyperparameter_schema['model_params'].items()}
            if "optimizer_params" in model_def.hyperparameter_schema:
                default_hparams['optimizer_params'] = {k: v['default'] for k, v in model_def.hyperparameter_schema['optimizer_params'].items()}
            trials.append(Trial(id=f"{model_name[:4]}_{uuid.uuid4().hex[:4]}", algorithm_name=model_name, hyperparameters=default_hparams))
        return trials

    def _setup_and_run_scheduler(self, trials: list, dataset_name: str):
        if not trials:
            self.append_log_message("ERROR: No trials were generated for the experiment.")
            self.update_button_states(running=False, paused=False)
            return
        dataset_def = AVAILABLE_DATASETS[dataset_name]
        adaptive_scheduler = SuccessiveHalvingScheduler(metric=dataset_def.performance_metric_name, increasing=True, min_epochs_per_rung=2, reduction_factor=2)
        scheduler = Scheduler(trials=trials, dataset_name=dataset_name, adaptive_scheduler=adaptive_scheduler, max_workers=4, enable_checkpointing=self.checkpoint_checkbox.isChecked())
        self.scheduler_thread = QThread()
        self.scheduler_runner = SchedulerRunner(scheduler)
        self.scheduler_runner.moveToThread(self.scheduler_thread)
        self.scheduler_runner.log_message.connect(self.append_log_message)
        self.scheduler_runner.trial_updated.connect(self.update_trial_ui)
        self.scheduler_runner.trial_profiled.connect(self.on_trial_profiled)
        self.scheduler_runner.experiment_finished.connect(self.on_experiment_finished)
        self.scheduler_runner.insight_generated.connect(self.add_insight)
        self.scheduler_thread.started.connect(self.scheduler_runner.run)
        self.update_throttle(self.throttle_slider.value())
        self.scheduler_thread.start()

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
            for trial_id, curve in self.plot_curve_map.items():
                pen = curve.opts['pen']
                color = pen.color()
                if trial_id == selected_trial_id:
                    color.setAlpha(255)
                    curve.setPen(pg.mkPen(color=color, width=4))
                    curve.setZValue(100)
                else:
                    color.setAlpha(30)
                    curve.setPen(pg.mkPen(color=color, width=1))
                    curve.setZValue(0)

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
        self.trials_table.setItem(row, 2, QTableWidgetItem(trial_data['status']))
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

        if metric_list:
            epochs, metrics = zip(*metric_list)
            self.plot_curve_map[trial_id].setData(epochs, metrics)

        if self.scheduler_runner:
            total_trials = len(self.scheduler_runner.scheduler.trials)
            if total_trials > 0:
                completed_statuses = {"COMPLETED", "PRUNED"}
                completed_trials = sum(1 for t in self.scheduler_runner.scheduler.trials.values() if t.status.value in completed_statuses)
                progress = int((completed_trials / total_trials) * 100)
                self.progress_bar.setValue(progress)

    def add_insight(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}] {message}")
        self.insights_list.addItem(item)
        self.insights_list.scrollToBottom()

    def append_log_message(self, message: str):
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def on_experiment_finished(self):
        self.append_log_message("INFO: Experiment finished.")
        self.update_button_states(running=False, paused=False)
        self.progress_bar.setValue(100)
        if self.scheduler_thread and self.scheduler_thread.isRunning():
            self.scheduler_thread.quit()
            self.scheduler_thread.wait()
        self.scheduler_runner = None

    def closeEvent(self, event):
        if self.scheduler_runner:
            self.scheduler_runner.stop()
        if self.scheduler_thread and self.scheduler_thread.isRunning():
            self.scheduler_thread.quit()
            self.scheduler_thread.wait()
        event.accept()


if __name__ == '__main__':
    # A simple test block to run and view the UI layout
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
