from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtWidgets import QInputDialog
from PyQt6.QtWidgets import QMessageBox

from ..registry import registry
from .hyperparameters import HyperparameterDialog
from .hyperparameters import HyperparameterViewerDialog
from .hyperparameters import SimpleRunDialog
from .hyperparameters import SpawnDialog


class DialogService:
    def __init__(self, parent):
        self.parent = parent

    def show_simple_run_dialog(self, selected_models_names):
        selected_model_defs = [registry.get_model(name) for name in selected_models_names]
        dialog = SimpleRunDialog(selected_model_defs, self.parent)
        result = dialog.exec()
        if result == QDialog.DialogCode.Rejected:
            return None, None

        custom_hparams = None
        if result == SimpleRunDialog.RunWithEdits:
            custom_hparams = dialog.get_hyperparameters()

        return result, custom_hparams

    def show_tuning_dialog(self, selected_models_names):
        selected_model_defs = [registry.get_model(name) for name in selected_models_names]
        dialog = HyperparameterDialog(selected_model_defs, self.parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return dialog.get_configuration()

    def show_hyperparameter_viewer(self, hparams):
        if not hparams:
            QMessageBox.information(
                self.parent, "Info", "No hyperparameters to display."
            )
            return
        dialog = HyperparameterViewerDialog(hparams, self.parent)
        dialog.exec()

    def show_spawn_dialog(self, source_trial_hparams):
        dialog = SpawnDialog(source_trial_hparams, self.parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_hyperparameters()
        return None

    def show_add_trials_dialog(self, num_new_models):
        num_trials, ok = QInputDialog.getInt(
            self.parent,
            "Add Trials",
            f"How many trials to generate for {num_new_models} new model(s)?",
            10,
            1,
            1000,
            1,
        )
        if ok:
            return num_trials
        return None

    def show_save_experiment_dialog(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self.parent, "Save Experiment", "", "SDE JSON Files (*.sde.json)"
        )
        return filepath

    def show_load_experiment_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self.parent, "Load Experiment", "", "SDE JSON Files (*.sde.json)"
        )
        return filepath

    def show_warning(self, title, message):
        QMessageBox.warning(self.parent, title, message)
