import sys
import uuid
import json
import itertools
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QSplitter,
    QPushButton, QSizePolicy, QComboBox, QLabel, QListWidget, QListWidgetItem,
    QFormLayout, QSlider, QCheckBox, QProgressBar, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import pyqtgraph as pg

# Import backend components
from sde.engine.scheduler import Scheduler
from sde.core.types import Trial
from sde.exploration.schedulers import SuccessiveHalvingScheduler
from sde.models import AVAILABLE_MODELS
from sde.challenges import AVAILABLE_DATASETS

class SchedulerRunner(QObject):
    """
    A QObject wrapper that runs the framework-agnostic Scheduler in a QThread
    and translates its custom signals into PyQt signals.
    """
    # PyQt signals that will be emitted from the UI thread
    log_message = pyqtSignal(str)
    trial_updated = pyqtSignal(dict)
    experiment_finished = pyqtSignal()
    insight_generated = pyqtSignal(str)

    def __init__(self, scheduler: Scheduler):
        super().__init__()
        self.scheduler = scheduler
        self._is_running = True

        # Connect the scheduler's custom signals to methods that emit PyQt signals
        self.scheduler.log_message.connect(self.log_message.emit)
        self.scheduler.trial_updated.connect(self.trial_updated.emit)
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

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget) # Changed to Horizontal

        # --- Left Pane: Experiment Setup ---
        setup_pane = QWidget()
        setup_pane.setMaximumWidth(350)
        setup_layout = QVBoxLayout(setup_pane)

        # --- Right Pane: Results ---
        results_pane = QWidget()
        results_layout = QVBoxLayout(results_pane)

        main_layout.addWidget(setup_pane)
        main_layout.addWidget(results_pane, 1) # Give results pane more space

        # --- Experiment Setup Group ---
        setup_group = QGroupBox("1. Experiment Setup")
        setup_form_layout = QFormLayout(setup_group)

        self.dataset_combo = QComboBox()
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.model_list.setMinimumHeight(150) # Ensure it has some space

        setup_form_layout.addRow("Dataset:", self.dataset_combo)
        setup_form_layout.addRow("Models:", self.model_list)

        # --- Execution Settings Group ---
        settings_group = QGroupBox("2. Execution Settings")
        settings_form_layout = QFormLayout(settings_group)

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

        execution_controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Experiment")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        execution_controls_layout.addWidget(self.start_button)
        execution_controls_layout.addWidget(self.pause_button)
        execution_controls_layout.addWidget(self.resume_button)

        # --- Assemble Left Pane ---
        setup_layout.addWidget(setup_group)
        setup_layout.addWidget(settings_group)
        setup_layout.addStretch(1)
        setup_layout.addWidget(self.progress_bar)
        setup_layout.addLayout(execution_controls_layout)

        # --- Connect Signals and Slots ---
        self.populate_datasets()
        self.dataset_combo.currentIndexChanged.connect(self.update_model_list)
        self.start_button.clicked.connect(self.start_experiment)
        self.pause_button.clicked.connect(self.pause_experiment)
        self.resume_button.clicked.connect(self.resume_experiment)
        self.throttle_slider.valueChanged.connect(self.update_throttle)

        self.update_model_list() # Initial population
        self.update_button_states(running=False, paused=False)

        # --- Main Content Splitter (in results_pane)---
        splitter = QSplitter(Qt.Orientation.Vertical)
        results_layout.addWidget(splitter)

        # --- Top Pane: Plot ---
        self.plot_widget = pg.PlotWidget()
        self.setup_plot()

        # --- Bottom Pane: Table and Log ---
        bottom_pane = QWidget()
        bottom_layout = QHBoxLayout(bottom_pane)

        self.trials_table = QTableWidget()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self.trials_table)
        bottom_splitter.addWidget(self.log_text_edit)
        bottom_splitter.setSizes([750, 450])
        bottom_layout.addWidget(bottom_splitter)

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
        if not selected_dataset_name:
            return

        dataset_def = AVAILABLE_DATASETS[selected_dataset_name]
        supported_models = [
            name for name, model_def in AVAILABLE_MODELS.items()
            if dataset_def.type in model_def.supported_dataset_types
        ]

        for model_name in supported_models:
            item = QListWidgetItem(model_name)
            # Make item checkable
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.model_list.addItem(item)

    def setup_plot(self):
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Real-Time Trial Performance", color="k", size="16pt")
        self.plot_widget.setLabel('left', 'Accuracy', color='k', **{'font-size': '12pt'})
        self.plot_widget.setLabel('bottom', 'Epoch', color='k', **{'font-size': '12pt'})
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()

    def setup_table(self):
        self.trials_table.setColumnCount(7) # Added Algorithm column
        self.trials_table.setHorizontalHeaderLabels([
            "Trial ID", "Algorithm", "Status", "Epoch", "Accuracy", "Loss", "Hyperparameters"
        ])
        header = self.trials_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.trials_table.setColumnWidth(0, 100)
        self.trials_table.setColumnWidth(1, 120)
        self.trials_table.setColumnWidth(2, 100)
        self.trials_table.setColumnWidth(3, 60)
        self.trials_table.setColumnWidth(4, 100)
        self.trials_table.setColumnWidth(5, 100)

    def update_button_states(self, running: bool, paused: bool):
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running and not paused)
        self.resume_button.setEnabled(running and paused)
        self.dataset_combo.setEnabled(not running)
        self.model_list.setEnabled(not running)
        self.checkpoint_checkbox.setEnabled(not running)

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

    def start_experiment(self):
        dataset_name, selected_models = self._get_experiment_settings()
        if not dataset_name or not selected_models:
            self.append_log_message("ERROR: Please select a dataset and at least one model.")
            return

        self.update_button_states(running=True, paused=False)
        self.append_log_message(f"INFO: Starting new experiment on '{dataset_name}' with models: {selected_models}")

        self.progress_bar.setValue(0)
        self._clear_previous_experiment()
        trials = self._create_trials(selected_models)
        self._setup_and_run_scheduler(trials, dataset_name)

    def _get_experiment_settings(self):
        dataset_name = self.dataset_combo.currentText()
        selected_models = [
            self.model_list.item(i).text()
            for i in range(self.model_list.count())
            if self.model_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        return dataset_name, selected_models

    def _clear_previous_experiment(self):
        self.trials_table.setRowCount(0)
        self.plot_widget.clear()
        self.trial_row_map.clear()
        self.plot_curve_map.clear()
        self.setup_plot() # Re-add legend, title, etc.

    def _create_trials(self, selected_models: list) -> list:
        trials = []
        for model_name in selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            # For simplicity, use default hyperparameters defined in the model's schema
            default_hparams = {}
            if "model_params" in model_def.hyperparameter_schema:
                default_hparams['model_params'] = {
                    k: v['default'] for k, v in model_def.hyperparameter_schema['model_params'].items()
                }
            if "optimizer_params" in model_def.hyperparameter_schema:
                default_hparams['optimizer_params'] = {
                    k: v['default'] for k, v in model_def.hyperparameter_schema['optimizer_params'].items()
                }

            trials.append(Trial(
                id=f"{model_name[:4]}_{uuid.uuid4().hex[:4]}",
                algorithm_name=model_name,
                hyperparameters=default_hparams
            ))
        return trials

    def _setup_and_run_scheduler(self, trials: list, dataset_name: str):
        dataset_def = AVAILABLE_DATASETS[dataset_name]
        adaptive_scheduler = SuccessiveHalvingScheduler(
            metric=dataset_def.performance_metric_name,
            increasing=True,  # Assuming higher metric is better
            min_epochs_per_rung=2,
            reduction_factor=2
        )

        scheduler = Scheduler(
            trials=trials,
            dataset_name=dataset_name,
            adaptive_scheduler=adaptive_scheduler,
            max_workers=4,  # TODO: Make this configurable in the UI
            enable_checkpointing=self.checkpoint_checkbox.isChecked()
        )

        self.scheduler_thread = QThread()
        self.scheduler_runner = SchedulerRunner(scheduler)
        self.scheduler_runner.moveToThread(self.scheduler_thread)

        # Connect signals from the runner to the UI
        self.scheduler_runner.log_message.connect(self.append_log_message)
        self.scheduler_runner.trial_updated.connect(self.update_trial_ui)
        self.scheduler_runner.experiment_finished.connect(self.on_experiment_finished)
        self.scheduler_runner.insight_generated.connect(self.append_log_message)

        # Start the thread, which will then start the scheduler
        self.scheduler_thread.started.connect(self.scheduler_runner.run)
        self.update_throttle(self.throttle_slider.value())
        self.scheduler_thread.start()

    def update_trial_ui(self, trial_data: dict):
        trial_id = trial_data['id']

        # -- Update Table --
        if trial_id not in self.trial_row_map:
            row_position = self.trials_table.rowCount()
            self.trials_table.insertRow(row_position)
            self.trial_row_map[trial_id] = row_position
            # Add a basic plot curve for the new trial
            color = pg.intColor(len(self.plot_curve_map), hues=9, values=1)
            pen = pg.mkPen(color=color, width=2)
            self.plot_curve_map[trial_id] = self.plot_widget.plot(
                [], [], name=f"{trial_data['algorithm_name']} ({trial_id})", pen=pen, symbol='o', symbolSize=6, symbolBrush=pen.color()
            )

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
        self.trials_table.setItem(row, 6, QTableWidgetItem(json.dumps(trial_data['hyperparameters'])))

        # -- Update Plot --
        if metric_list:
            epochs, metrics = zip(*metric_list)
            self.plot_curve_map[trial_id].setData(epochs, metrics)

        # -- Update Progress Bar --
        if self.scheduler_runner:
            total_trials = len(self.scheduler_runner.scheduler.trials)
            if total_trials > 0:
                completed_statuses = {"COMPLETED", "PRUNED"}
                completed_trials = sum(1 for t in self.scheduler_runner.scheduler.trials.values() if t.status.value in completed_statuses)
                progress = int((completed_trials / total_trials) * 100)
                self.progress_bar.setValue(progress)

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
        """Ensure the scheduler thread is stopped cleanly on exit."""
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
