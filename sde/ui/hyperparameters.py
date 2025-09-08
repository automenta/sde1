from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QDialogButtonBox
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QFormLayout
from PyQt6.QtWidgets import QGroupBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QSpinBox
from PyQt6.QtWidgets import QTreeWidget
from PyQt6.QtWidgets import QTreeWidgetItem
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from sde.models.types import ModelDefinition


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


class HyperparameterDialog(QDialog):
    """A dialog for configuring a hyperparameter tuning experiment.
    """

    def __init__(self, models: list[ModelDefinition], parent=None):
        super().__init__(parent)
        self.models = models
        self.config = {}
        self.param_widgets = {}

        self.setWindowTitle("Configure Hyperparameter Tuning")
        self.setMinimumSize(600, 500)

        main_layout = QVBoxLayout(self)

        # --- Adaptive Scheduler Selection ---
        ada_sched_group = QGroupBox("Adaptive Scheduling Strategy")
        ada_sched_layout = QFormLayout(ada_sched_group)
        self.ada_sched_combo = QComboBox()
        self.ada_sched_combo.addItems(["Successive Halving", "Hyperband"])
        self.max_epochs_spinbox = QSpinBox()
        self.max_epochs_spinbox.setRange(10, 1000)
        self.max_epochs_spinbox.setValue(81)
        self.max_epochs_label = QLabel("Max Epochs per Trial:")
        ada_sched_layout.addRow("Scheduler:", self.ada_sched_combo)
        ada_sched_layout.addRow(self.max_epochs_label, self.max_epochs_spinbox)
        self.ada_sched_combo.currentTextChanged.connect(self._update_scheduler_widgets)

        # --- Hyperparameter Generation Strategy ---
        strategy_group = QGroupBox("Hyperparameter Generation Strategy")
        strategy_layout = QFormLayout(strategy_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Grid Search", "Random Search"])
        self.num_trials_spinbox = QSpinBox()
        self.num_trials_spinbox.setRange(1, 10000)
        self.num_trials_label = QLabel("Number of Trials:")
        strategy_layout.addRow("Strategy:", self.strategy_combo)
        strategy_layout.addRow(self.num_trials_label, self.num_trials_spinbox)
        self.strategy_combo.currentTextChanged.connect(self._update_strategy_label)
        self._update_strategy_label(self.strategy_combo.currentText())
        self._update_scheduler_widgets(self.ada_sched_combo.currentText())

        # --- Parameters ---
        params_group = QGroupBox("Hyperparameter Space")
        params_main_layout = QVBoxLayout(params_group)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.params_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        params_main_layout.addWidget(scroll_area)

        # --- Dialog Buttons ---
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(ada_sched_group)
        main_layout.addWidget(strategy_group)
        main_layout.addWidget(params_group, 1)  # Give more space to params
        main_layout.addWidget(button_box)

        self._populate_hyperparameters()

    def _update_scheduler_widgets(self, scheduler: str):
        is_hyperband = scheduler == "Hyperband"
        self.max_epochs_label.setVisible(is_hyperband)
        self.max_epochs_spinbox.setVisible(is_hyperband)

    def _update_strategy_label(self, strategy: str):
        if strategy == "Grid Search":
            self.num_trials_label.setText("Steps per Hyperparameter:")
            self.num_trials_spinbox.setValue(3)
        else:  # Random Search
            self.num_trials_label.setText("Total Number of Trials:")
            self.num_trials_spinbox.setValue(20)

    def _populate_hyperparameters(self):
        """Dynamically create widgets for each model's hyperparameters."""
        for model_def in self.models:
            model_group = QGroupBox(model_def.name)
            model_layout = QFormLayout(model_group)
            self.param_widgets[model_def.name] = {}

            schema = model_def.hyperparameter_schema
            for param_type in ["model_params", "optimizer_params"]:
                if param_type not in schema:
                    continue

                self.param_widgets[model_def.name][param_type] = {}
                for param_name, properties in schema[param_type].items():
                    self.param_widgets[model_def.name][param_type][param_name] = {}

                    # Create widgets based on parameter type
                    if properties["type"] == "float":
                        min_val, max_val = properties["min"], properties["max"]

                        min_box = QDoubleSpinBox()
                        min_box.setRange(min_val, max_val)
                        min_box.setSingleStep((max_val - min_val) / 100)
                        min_box.setValue(min_val)
                        min_box.setDecimals(6)

                        max_box = QDoubleSpinBox()
                        max_box.setRange(min_val, max_val)
                        max_box.setSingleStep((max_val - min_val) / 100)
                        max_box.setValue(max_val)
                        max_box.setDecimals(6)

                        scale_combo = QComboBox()
                        scale_combo.addItems(["Linear", "Log"])

                        widget_layout = QHBoxLayout()
                        widget_layout.addWidget(min_box)
                        widget_layout.addWidget(QLabel("to"))
                        widget_layout.addWidget(max_box)
                        widget_layout.addWidget(QLabel("Scale:"))
                        widget_layout.addWidget(scale_combo)

                        model_layout.addRow(f"{param_name}:", widget_layout)

                        self.param_widgets[model_def.name][param_type][param_name][
                            "min"
                        ] = min_box
                        self.param_widgets[model_def.name][param_type][param_name][
                            "max"
                        ] = max_box
                        self.param_widgets[model_def.name][param_type][param_name][
                            "scale"
                        ] = scale_combo

            self.params_layout.addWidget(model_group)

    def get_configuration(self):
        """Constructs the configuration dictionary from the UI widgets."""
        self.config["adaptive_scheduler"] = self.ada_sched_combo.currentText()
        if self.config["adaptive_scheduler"] == "Hyperband":
            self.config["max_epochs"] = self.max_epochs_spinbox.value()

        self.config["hparam_strategy"] = self.strategy_combo.currentText()
        self.config["num_trials"] = self.num_trials_spinbox.value()
        self.config["models"] = {}

        for model_name, param_types in self.param_widgets.items():
            self.config["models"][model_name] = {}
            for param_type, params in param_types.items():
                self.config["models"][model_name][param_type] = {}
                for param_name, widgets in params.items():
                    self.config["models"][model_name][param_type][param_name] = {
                        "min": widgets["min"].value(),
                        "max": widgets["max"].value(),
                        "scale": widgets["scale"].currentText().lower(),
                    }
        return self.config

    def accept(self):
        """Overrides the default accept to store the config."""
        self.config = self.get_configuration()
        super().accept()


class SpawnDialog(QDialog):
    """A dialog to edit hyperparameters before spawning a new trial."""

    def __init__(self, hparams: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spawn Similar Trial")
        self.setLayout(QVBoxLayout())
        self.resize(450, 350)

        self.editor_widgets = {}
        form_layout = QFormLayout()

        for key, value in hparams.items():
            if isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setDecimals(6)
                widget.setValue(value)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setValue(value)
            else:
                # Fallback to a line edit for strings or other types
                widget = QLineEdit(str(value))

            self.editor_widgets[key] = widget
            form_layout.addRow(key, widget)

        self.layout().addLayout(form_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout().addWidget(button_box)

    def get_hyperparameters(self) -> dict:
        """Constructs a new hyperparameter dictionary from the editor widgets."""
        new_hparams = {}
        for key, widget in self.editor_widgets.items():
            if isinstance(widget, QDoubleSpinBox) or isinstance(widget, QSpinBox):
                new_hparams[key] = widget.value()
            else:
                # Attempt to convert back to original type if possible
                try:
                    original_type = type(widget.text())
                    new_hparams[key] = original_type(widget.text())
                except (ValueError, TypeError):
                    new_hparams[key] = widget.text() # Keep as string if conversion fails
        return new_hparams
