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
from PyQt6.QtWidgets import QLineEdit
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
    start_experiment_requested = pyqtSignal(dict)
    add_models_requested = pyqtSignal()
    pause_run_requested = pyqtSignal()
    resume_run_requested = pyqtSignal()
    stop_run_requested = pyqtSignal()
    save_run_requested = pyqtSignal()
    load_run_requested = pyqtSignal()
    edit_hparams_requested = pyqtSignal()
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
        setup_layout = QVBoxLayout(self.setup_group)
        setup_form_layout = QFormLayout()

        self.dataset_combo = QComboBox()
        setup_form_layout.addRow("Dataset:", self.dataset_combo)

        self.model_search_input = QLineEdit()
        self.model_search_input.setPlaceholderText("Filter models by name...")
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.model_list.setMinimumHeight(150)
        self.model_list.setToolTip(
            "Select one or more models to include in the experiment.\n"
            "Models compatible with the selected dataset will appear here."
        )
        setup_layout.addLayout(setup_form_layout)
        setup_layout.addWidget(QLabel("Models:"))
        setup_layout.addWidget(self.model_search_input)
        setup_layout.addWidget(self.model_list)

        self.model_details_group = QGroupBox("Model Details")
        model_details_layout = QVBoxLayout(self.model_details_group)
        self.model_details_text = QTextEdit()
        self.model_details_text.setReadOnly(True)
        self.model_details_text.setPlaceholderText("Click on a model to see its details.")
        model_details_layout.addWidget(self.model_details_text)
        self.model_details_group.setVisible(False)  # Initially hidden
        setup_layout.addWidget(self.model_details_group)


        # --- Compute Settings Group ---
        self.settings_group = QGroupBox("2. Compute Settings")
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
        self.worker_count_spinbox.setToolTip(
            "Number of parallel CPU workers to run computation tasks (like training or evaluation).\n"
            "Set this to the number of physical CPU cores for maximum throughput."
        )
        settings_form_layout.addRow("Parallel Workers:", self.worker_count_spinbox)

        # Timeout per Work Unit
        self.timeout_spinbox = QSpinBox()
        self.timeout_spinbox.setMinimum(1)
        self.timeout_spinbox.setMaximum(3600)
        self.timeout_spinbox.setValue(300)
        self.timeout_spinbox.setToolTip(
            "Maximum time (in seconds) allowed for a single Work Unit (e.g., one training epoch).\n"
            "If a unit exceeds this, it's marked as failed.\n"
            "This prevents a single stalled trial from halting the entire experiment."
        )
        settings_form_layout.addRow("Work Unit Timeout (s):", self.timeout_spinbox)
        self.checkpoint_checkbox = QCheckBox("Enable Checkpointing")
        self.checkpoint_checkbox.setChecked(False)
        settings_form_layout.addRow(self.checkpoint_checkbox)

        # --- Run Configuration Group ---
        run_config_group = QGroupBox("3. Run Configuration")
        run_config_form_layout = QFormLayout(run_config_group)
        self.run_mode_combo = QComboBox()
        self.run_mode_combo.addItems(["Simple", "Tune Hyperparameters"])
        self.trials_per_algo_spinbox = QSpinBox()
        self.trials_per_algo_spinbox.setMinimum(1)
        self.trials_per_algo_spinbox.setMaximum(1000)
        self.trials_per_algo_spinbox.setValue(10)
        self.trials_per_algo_spinbox.setToolTip(
            "For 'Simple' mode, this is the number of random hyperparameter configurations to generate for each selected algorithm.\n"
            "More trials increase the chance of finding a good configuration, but require more computation."
        )
        run_config_form_layout.addRow("Run Mode:", self.run_mode_combo)
        run_config_form_layout.addRow("Trials per Algorithm:", self.trials_per_algo_spinbox)

        self.edit_hparams_button = QPushButton("Edit Hyperparameters...")
        self.edit_hparams_button.setToolTip(
            "For 'Simple' mode, this allows you to view and override the default\n"
            "hyperparameter sampling space for the selected models."
        )
        run_config_form_layout.addRow(self.edit_hparams_button)


        # --- Experiment Controls ---
        controls_group = QGroupBox("4. Experiment Controls")
        controls_layout = QVBoxLayout(controls_group)
        self.start_experiment_button = QPushButton("Start Experiment")
        self.add_models_button = QPushButton("Add Selected Models to Run")
        self.add_models_button.setToolTip(
            "While a run is active, select new models from the list above\n"
            "and click here to add them to the experiment."
        )
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.stop_button = QPushButton("Stop")
        self.save_button = QPushButton("Save Run")
        self.load_button = QPushButton("Load Run")

        start_layout = QHBoxLayout()
        start_layout.addWidget(self.start_experiment_button)
        mid_run_layout = QHBoxLayout()
        mid_run_layout.addWidget(self.add_models_button)
        pause_resume_layout = QHBoxLayout()
        pause_resume_layout.addWidget(self.pause_button)
        pause_resume_layout.addWidget(self.resume_button)
        pause_resume_layout.addWidget(self.stop_button)
        persistence_layout = QHBoxLayout()
        persistence_layout.addWidget(self.save_button)
        persistence_layout.addWidget(self.load_button)

        controls_layout.addLayout(start_layout)
        controls_layout.addLayout(mid_run_layout)
        controls_layout.addLayout(pause_resume_layout)
        controls_layout.addLayout(persistence_layout)

        # Worker Throttle
        self.throttle_slider = QSlider(Qt.Orientation.Horizontal)
        self.throttle_slider.setRange(1, 100)
        self.throttle_slider.setValue(100)
        self.throttle_label = QLabel("100%")
        throttle_widget = QWidget()
        throttle_layout = QHBoxLayout(throttle_widget)
        throttle_layout.addWidget(self.throttle_slider)
        throttle_layout.addWidget(self.throttle_label)
        throttle_layout.setContentsMargins(0, 0, 0, 0)
        throttle_form_layout = QFormLayout()
        throttle_form_layout.addRow("Worker Throttle:", throttle_widget)
        controls_layout.addLayout(throttle_form_layout)


        # --- Algorithm Management Group ---
        self.algorithms_group = QGroupBox("5. Active Algorithms")
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
        main_layout.addWidget(run_config_group)
        main_layout.addWidget(controls_group)
        main_layout.addWidget(self.algorithms_group)
        main_layout.addStretch(1)
        main_layout.addWidget(self.progress_bar)

    def _connect_signals(self):
        """Connects internal UI signals to the pane's public signals."""
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        self.model_search_input.textChanged.connect(self._update_model_filter)
        self.model_list.itemSelectionChanged.connect(self._on_model_selection_changed)
        self.start_experiment_button.clicked.connect(self._on_start_experiment)
        self.add_models_button.clicked.connect(self.add_models_requested)
        self.pause_button.clicked.connect(self.pause_run_requested)
        self.resume_button.clicked.connect(self.resume_run_requested)
        self.stop_button.clicked.connect(self.stop_run_requested)
        self.save_button.clicked.connect(self.save_run_requested)
        self.load_button.clicked.connect(self.load_run_requested)
        self.edit_hparams_button.clicked.connect(self.edit_hparams_requested)
        self.throttle_slider.valueChanged.connect(self.throttle_changed)
        self.throttle_slider.valueChanged.connect(lambda v: self.throttle_label.setText(f"{v}%"))

    def _on_dataset_changed(self, index: int):
        """Handles the combo box's index change and emits the dataset name."""
        dataset_name = self.dataset_combo.itemText(index) if index >= 0 else ""
        self.dataset_changed.emit(dataset_name)
        # Also update the model list directly when the dataset changes
        self.update_model_list(dataset_name)
        # Clear the filter when the dataset changes
        self.model_search_input.clear()
        self._on_model_selection_changed() # Clear details pane

    def _on_model_selection_changed(self):
        """Updates the model details view when a model is selected."""
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            self.model_details_group.setVisible(False)
            self.model_details_text.clear()
            return

        # For simplicity, show details for the first selected item
        model_name = selected_items[0].text()
        model_def = AVAILABLE_MODELS.get(model_name)

        if not model_def:
            self.model_details_text.setText(f"Could not find details for '{model_name}'.")
            self.model_details_group.setVisible(True)
            return

        # Build an HTML string for display
        details_html = f"<h3>{model_def.get('name', model_name)}</h3>"
        details_html += f"<p><i>{model_def.get('description', 'No description available.')}</i></p>"
        details_html += f"<b>Architecture Type:</b> {model_def.get('architecture_type', 'N/A')}<br>"

        schema = model_def.get('hyperparameter_schema', {})
        if schema:
            details_html += "<b>Hyperparameters:</b><ul>"
            for group, params in schema.items():
                for param_name, properties in params.items():
                    details_html += f"<li><b>{param_name}</b>: "
                    if 'values' in properties:
                        details_html += f"Categorical {properties['values']}"
                    else:
                        scale = f" ({properties.get('scale', 'linear')} scale)"
                        details_html += f"Range [{properties.get('min', 'N/A')}, {properties.get('max', 'N/A')}]"
                        details_html += f"<small>{scale}</small>"
                    details_html += "</li>"
            details_html += "</ul>"

        self.model_details_text.setHtml(details_html)
        self.model_details_group.setVisible(True)


    def _update_model_filter(self):
        """Filters the model list based on the search input text."""
        filter_text = self.model_search_input.text().lower()
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            item_text = item.text().lower()
            # The item is hidden if the filter text is not a substring of the item text
            item.setHidden(filter_text not in item_text)

    def _on_start_experiment(self):
        """Gathers settings and emits the start signal."""
        settings = self.get_execution_settings()
        self.start_experiment_requested.emit(settings)

    def get_execution_settings(self) -> dict:
        """Gathers all execution-related settings from the UI controls."""
        return {
            "run_mode": self.run_mode_combo.currentText(),
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
        self.start_experiment_button.setEnabled(can_start)

        self.pause_button.setEnabled("PAUSE_RUN" in global_actions)
        self.resume_button.setEnabled("RESUME_RUN" in global_actions)
        self.stop_button.setEnabled("STOP_RUN" in global_actions)

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
        is_simple_mode = self.run_mode_combo.currentText() == "Simple"
        can_edit_hparams = (
            status == "DEFINING" and is_simple_mode and self.model_list.selectedItems()
        )
        self.edit_hparams_button.setEnabled(can_edit_hparams)

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
