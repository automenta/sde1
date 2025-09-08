import os

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QFormLayout
from PyQt6.QtWidgets import QGroupBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QHeaderView
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QListWidget
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtWidgets import QProgressBar
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QSlider
from PyQt6.QtWidgets import QSpinBox
from PyQt6.QtWidgets import QTableWidget
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from ..challenges import AVAILABLE_DATASETS
from ..models import AVAILABLE_MODELS


class SetupPane(QWidget):
    """The left-hand pane for experiment setup and controls.
    It encapsulates all the widgets and logic for configuring and
    controlling an experiment run.
    """

    # Signals to communicate user actions to the main window
    start_simple_run_requested = pyqtSignal(dict)
    tune_run_requested = pyqtSignal()
    add_models_requested = pyqtSignal()
    pause_run_requested = pyqtSignal()
    resume_run_requested = pyqtSignal()
    save_run_requested = pyqtSignal()
    load_run_requested = pyqtSignal()
    dataset_changed = pyqtSignal(str)
    throttle_changed = pyqtSignal(int)
    remove_algorithm_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(350)
        self._init_ui()
        self._connect_signals()
        self.populate_datasets()

    def _init_ui(self):
        """Initializes the UI layout and sub-components."""
        main_layout = QVBoxLayout(self)

        # --- Experiment Setup Group ---
        self.setup_group = QGroupBox("1. Experiment Setup")
        setup_form_layout = QFormLayout(self.setup_group)
        self.dataset_combo = QComboBox()
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.model_list.setMinimumHeight(150)
        self.model_list.setToolTip(
            "Select one or more models to include in the experiment.\n"
            "Models compatible with the selected dataset will appear here."
        )
        setup_form_layout.addRow("Dataset:", self.dataset_combo)
        setup_form_layout.addRow("Models:", self.model_list)

        # --- Execution Settings Group ---
        self.settings_group = QGroupBox("2. Execution Settings")
        settings_form_layout = QFormLayout(self.settings_group)

        # Worker Count
        self.worker_count_spinbox = QSpinBox()
        self.worker_count_spinbox.setMinimum(1)
        # Set max to physical CPU count, or a reasonable default if that fails
        try:
            max_workers = os.cpu_count() or 4
        except NotImplementedError:
            max_workers = 4
        self.worker_count_spinbox.setMaximum(max_workers)
        self.worker_count_spinbox.setValue(max_workers)
        self.worker_count_spinbox.setToolTip("Number of parallel processes for computation.")
        settings_form_layout.addRow("Parallel Workers:", self.worker_count_spinbox)

        # Trials per Algorithm
        self.trials_per_algo_spinbox = QSpinBox()
        self.trials_per_algo_spinbox.setMinimum(1)
        self.trials_per_algo_spinbox.setMaximum(1000)
        self.trials_per_algo_spinbox.setValue(10)
        self.trials_per_algo_spinbox.setToolTip("Number of random hyperparameter sets to generate per algorithm.")
        settings_form_layout.addRow("Trials per Algorithm:", self.trials_per_algo_spinbox)

        # Timeout per Work Unit
        self.timeout_spinbox = QSpinBox()
        self.timeout_spinbox.setMinimum(1)
        self.timeout_spinbox.setMaximum(3600)
        self.timeout_spinbox.setValue(300)
        self.timeout_spinbox.setToolTip(
            "Maximum time (in seconds) to wait for a single work unit (e.g., one epoch)\n"
            "before considering it failed. Prevents the engine from freezing on a stuck trial."
        )
        settings_form_layout.addRow("Work Unit Timeout (s):", self.timeout_spinbox)

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
        self.tune_button = QPushButton("Tune Hyperparameters...")
        self.add_models_button = QPushButton("Add Selected Models to Run")
        self.add_models_button.setToolTip(
            "While a run is active, select new models from the list above\n"
            "and click here to add them to the experiment."
        )
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.save_button = QPushButton("Save Run")
        self.load_button = QPushButton("Load Run")

        simple_run_layout = QHBoxLayout()
        simple_run_layout.addWidget(self.start_button)
        advanced_run_layout = QHBoxLayout()
        advanced_run_layout.addWidget(self.tune_button)
        mid_run_layout = QHBoxLayout()
        mid_run_layout.addWidget(self.add_models_button)
        pause_resume_layout = QHBoxLayout()
        pause_resume_layout.addWidget(self.pause_button)
        pause_resume_layout.addWidget(self.resume_button)
        persistence_layout = QHBoxLayout()
        persistence_layout.addWidget(self.save_button)
        persistence_layout.addWidget(self.load_button)

        controls_layout.addLayout(simple_run_layout)
        controls_layout.addLayout(advanced_run_layout)
        controls_layout.addLayout(mid_run_layout)
        controls_layout.addLayout(pause_resume_layout)
        controls_layout.addLayout(persistence_layout)

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

        # --- Assemble Pane ---
        main_layout.addWidget(self.setup_group)
        main_layout.addWidget(self.settings_group)
        main_layout.addWidget(controls_group)
        main_layout.addWidget(self.algorithms_group)
        main_layout.addStretch(1)
        main_layout.addWidget(self.progress_bar)

    def _connect_signals(self):
        """Connects internal UI signals to the pane's public signals."""
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        self.start_button.clicked.connect(self._on_start_simple_run)
        self.tune_button.clicked.connect(self.tune_run_requested)
        self.add_models_button.clicked.connect(self.add_models_requested)
        self.pause_button.clicked.connect(self.pause_run_requested)
        self.resume_button.clicked.connect(self.resume_run_requested)
        self.save_button.clicked.connect(self.save_run_requested)
        self.load_button.clicked.connect(self.load_run_requested)
        self.throttle_slider.valueChanged.connect(self.throttle_changed)
        self.throttle_slider.valueChanged.connect(lambda v: self.throttle_label.setText(f"{v}%"))

    def _on_dataset_changed(self, index: int):
        """Handles the combo box's index change and emits the dataset name."""
        dataset_name = self.dataset_combo.itemText(index) if index >= 0 else ""
        self.dataset_changed.emit(dataset_name)
        # Also update the model list directly when the dataset changes
        self.update_model_list(dataset_name)

    def _on_start_simple_run(self):
        """Gathers settings and emits the start signal."""
        settings = self.get_execution_settings()
        self.start_simple_run_requested.emit(settings)

    def get_execution_settings(self) -> dict:
        """Gathers all execution-related settings from the UI controls."""
        return {
            "num_workers": self.worker_count_spinbox.value(),
            "num_trials_per_algo": self.trials_per_algo_spinbox.value(),
            "enable_checkpointing": self.checkpoint_checkbox.isChecked(),
            "work_unit_timeout_seconds": self.timeout_spinbox.value(),
        }

    # --- Public Methods to Update UI State ---

    def populate_datasets(self):
        self.dataset_combo.addItems(AVAILABLE_DATASETS.keys())

    def update_model_list(self, selected_dataset_name: str):
        self.model_list.clear()
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
            self.model_list.addItem(item)

    def get_experiment_settings(self):
        dataset_name = self.dataset_combo.currentText()
        selected_models = [item.text() for item in self.model_list.selectedItems()]
        return dataset_name, selected_models

    def get_newly_selected_models(self, existing_algo_names: set):
        all_selected_models = {item.text() for item in self.model_list.selectedItems()}
        return all_selected_models - existing_algo_names

    def update_button_states(self, valid_actions: dict, status: str, has_challenge: bool):
        global_actions = valid_actions.get("global", [])

        can_start = "START_RUN" in global_actions
        self.start_button.setEnabled(can_start)
        self.tune_button.setEnabled(can_start)

        self.pause_button.setEnabled("PAUSE_RUN" in global_actions)
        self.resume_button.setEnabled("RESUME_RUN" in global_actions)

        # Persistence buttons
        can_save = status in ["RUNNING", "PAUSED", "COMPLETED"]
        self.save_button.setEnabled(can_save)
        self.load_button.setEnabled(status == "DEFINING") # Can only load when not running

        # Mid-run actions
        can_add_models = "ADD_ALGORITHM" in global_actions and status in ["RUNNING", "PAUSED"]
        self.add_models_button.setEnabled(can_add_models)

        self.dataset_combo.setEnabled(not has_challenge)
        self.model_list.setEnabled("ADD_ALGORITHM" in global_actions)
        self.settings_group.setEnabled(status == "DEFINING")

    def update_algorithm_table(self, algorithms: dict, valid_actions: dict):
        self.algorithms_table.setRowCount(0)
        algo_actions = valid_actions.get("algorithms", {})

        for algo_id, algo_data in algorithms.items():
            row_position = self.algorithms_table.rowCount()
            self.algorithms_table.insertRow(row_position)
            self.algorithms_table.setItem(row_position, 0, QListWidgetItem(algo_data.name))

            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            remove_button = QPushButton("Remove")
            remove_button.setEnabled("REMOVE_ALGORITHM" in algo_actions.get(algo_id, []))
            remove_button.clicked.connect(
                lambda _, a_id=algo_id: self.remove_algorithm_requested.emit(a_id)
            )
            actions_layout.addWidget(remove_button)
            self.algorithms_table.setCellWidget(row_position, 1, actions_widget)

    def update_progress_bar(self, progress: int):
        self.progress_bar.setValue(progress)

    def clear_algorithms_table(self):
        """Clears the algorithm table."""
        self.algorithms_table.setRowCount(0)
