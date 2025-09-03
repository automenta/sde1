import itertools
import random
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea, QWidget, QDialogButtonBox
)
from PyQt6.QtCore import Qt

from sde.models.types import ModelDefinition

class HyperparameterDialog(QDialog):
    """
    A dialog for configuring a hyperparameter tuning experiment.
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
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(ada_sched_group)
        main_layout.addWidget(strategy_group)
        main_layout.addWidget(params_group, 1) # Give more space to params
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
        else: # Random Search
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
                    if properties['type'] == 'float':
                        min_val, max_val = properties['min'], properties['max']

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

                        self.param_widgets[model_def.name][param_type][param_name]['min'] = min_box
                        self.param_widgets[model_def.name][param_type][param_name]['max'] = max_box
                        self.param_widgets[model_def.name][param_type][param_name]['scale'] = scale_combo

            self.params_layout.addWidget(model_group)

    def get_configuration(self):
        """Constructs the configuration dictionary from the UI widgets."""
        self.config['adaptive_scheduler'] = self.ada_sched_combo.currentText()
        if self.config['adaptive_scheduler'] == 'Hyperband':
            self.config['max_epochs'] = self.max_epochs_spinbox.value()

        self.config['hparam_strategy'] = self.strategy_combo.currentText()
        self.config['num_trials'] = self.num_trials_spinbox.value()
        self.config['models'] = {}

        for model_name, param_types in self.param_widgets.items():
            self.config['models'][model_name] = {}
            for param_type, params in param_types.items():
                self.config['models'][model_name][param_type] = {}
                for param_name, widgets in params.items():
                    self.config['models'][model_name][param_type][param_name] = {
                        'min': widgets['min'].value(),
                        'max': widgets['max'].value(),
                        'scale': widgets['scale'].currentText().lower()
                    }
        return self.config

    def accept(self):
        """Overrides the default accept to store the config."""
        self.config = self.get_configuration()
        super().accept()
