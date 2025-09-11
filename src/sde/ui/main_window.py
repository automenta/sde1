import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QInputDialog
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QProgressDialog
from PyQt6.QtWidgets import QWidget

from ..challenges import AVAILABLE_DATASETS

# Import backend and UI components
from ..core.actions import ActionType
from ..engine.orchestrator import ExperimentOrchestrator
from ..models import AVAILABLE_MODELS
from .hyperparameters import HyperparameterDialog
from .hyperparameters import HyperparameterViewerDialog
from .hyperparameters import SimpleRunDialog
from .hyperparameters import SpawnDialog
from .results_pane import ResultsPane
from .setup_pane import SetupPane
from .view_model import ExperimentViewModel


class MainWindow(QMainWindow):
    """The main window for the Scientific Discovery Engine UI. It orchestrates the
    SetupPane (left) and ResultsPane (right), and manages the ViewModel and
    the backend ExperimentOrchestrator.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Discovery Engine")
        self.setGeometry(100, 100, 1400, 900)

        # --- Backend and ViewModel ---
        self.orchestrator = ExperimentOrchestrator()
        self.view_model = ExperimentViewModel(self.style())

        self._init_ui()
        self._connect_signals()

        self.append_log_message({
            "level": "INFO",
            "message": "UI Initialized. Configure your experiment and click 'Start'."
        })

    def _init_ui(self):
        """Initializes the main UI layout and sub-components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.setup_pane = SetupPane()
        self.results_pane = ResultsPane()

        main_layout.addWidget(self.setup_pane)
        main_layout.addWidget(self.results_pane, 1)

        # --- Progress Dialog for Long Operations ---
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setMinimum(0)
        self.progress_dialog.setMaximum(0) # Makes it an indeterminate progress bar

    def _connect_signals(self):
        """Connects all UI signals to their corresponding slots."""
        # Backend signals -> Main Window
        self.orchestrator.log_message.connect(self.append_log_message)
        self.orchestrator.state_changed.connect(self.on_state_changed)
        self.orchestrator.operation_started.connect(self.on_operation_started)
        self.orchestrator.operation_finished.connect(self.on_operation_finished)

        # Setup Pane -> Main Window
        self.setup_pane.start_experiment_requested.connect(self.start_experiment)
        self.setup_pane.add_models_requested.connect(self.add_models_to_run)
        self.setup_pane.pause_run_requested.connect(self.pause_experiment)
        self.setup_pane.resume_run_requested.connect(self.resume_experiment)
        self.setup_pane.stop_run_requested.connect(self.stop_experiment)
        self.setup_pane.save_run_requested.connect(self.save_experiment)
        self.setup_pane.load_run_requested.connect(self.load_experiment)
        self.setup_pane.throttle_changed.connect(self.update_throttle)
        self.setup_pane.remove_algorithm_requested.connect(self.remove_algorithm)

        # Results Pane -> Main Window
        self.results_pane.trial_selected.connect(self.on_trial_selected)
        self.results_pane.trial_double_clicked.connect(self.on_trial_double_clicked)
        self.results_pane.insight_selected.connect(self.on_insight_selected)
        self.results_pane.prune_trial_requested.connect(self.prune_trial)
        self.results_pane.prioritize_trial_requested.connect(self.prioritize_trial)
        self.results_pane.spawn_trial_requested.connect(self.spawn_trial)
        self.results_pane.refresh_requested.connect(self.request_state_update)


    # --- Major Action Handlers ---

    def on_state_changed(self, state: dict):
        """The central handler for all state updates from the Orchestrator.
        It delegates state processing to the ViewModel and then triggers a UI refresh.
        """
        self.view_model.update_state(state)
        self._update_all_widgets()

    def start_experiment(self, settings: dict):
        """Starts an experiment based on the mode selected in the SetupPane."""
        run_mode = settings.get("run_mode")
        if run_mode == "Simple":
            self.start_simple_experiment(settings)
        elif run_mode == "Tune Hyperparameters":
            self.open_tuning_dialog(settings)
        else:
            self.append_log_message({"level": "ERROR", "message": f"Unknown run mode: {run_mode}"})

    def start_simple_experiment(self, settings: dict):
        """Handles the 'Simple' run mode by showing the SimpleRunDialog."""
        dataset_name, selected_models_names = self.setup_pane.get_experiment_settings()
        if not dataset_name or not selected_models_names:
            QMessageBox.warning(self, "Missing Information", "Please select a dataset and at least one model.")
            return

        selected_model_defs = [AVAILABLE_MODELS[name] for name in selected_models_names]
        dialog = SimpleRunDialog(selected_model_defs, self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Rejected:
            self.append_log_message({"level": "INFO", "message": "Experiment start cancelled by user."})
            return

        # --- Proceed with experiment setup ---
        self.append_log_message({
            "level": "INFO",
            "message": f"Configuring experiment on '{dataset_name}' with models: {selected_models_names}"
        })
        self._clear_previous_experiment()

        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.orchestrator.dispatch(
            ActionType.SET_CHALLENGE, {"name": dataset_name, "type": challenge_def.type}
        )

        custom_hparams = {}
        if result == SimpleRunDialog.RunWithEdits:
            custom_hparams = dialog.get_hyperparameters()
            self.append_log_message({"level": "INFO", "message": "Starting run with custom hyperparameters."})
        else: # RunWithDefaults
            self.append_log_message({"level": "INFO", "message": "Starting run with default hyperparameters."})


        for model_name in selected_models_names:
            model_def = AVAILABLE_MODELS[model_name]
            # For a simple run, the parameter space IS the set of single values.
            # The backend will create a single trial from this.
            param_space = {}
            # Use custom hparams if they exist for this model
            if model_name in custom_hparams:
                 param_space = custom_hparams[model_name]
            else:
                # Otherwise, extract defaults from the schema
                for param_type, params in model_def.hyperparameter_schema.items():
                    for param_name, properties in params.items():
                        param_space[param_name] = properties.get("default")

            self.orchestrator.dispatch(
                ActionType.ADD_ALGORITHM, {"name": model_name, "parameter_space": param_space, "is_simple_run": True}
            )

        self.orchestrator.dispatch(ActionType.START_RUN, settings)

    def open_tuning_dialog(self, settings: dict):
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
        self.append_log_message({
            "level": "INFO",
            "message": f"Configuring tuning experiment with scheduler '{config['adaptive_scheduler']}'."
        })
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

        self.append_log_message({
            "level": "INFO",
            "message": f"Starting run. The '{config['adaptive_scheduler']}' policy will now generate trials."
        })
        # Get the latest execution settings from the pane
        self.orchestrator.dispatch(ActionType.START_RUN, settings)

    # --- UI Update and State Management ---

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
        self.setup_pane.update_progress_bar(self.view_model.progress)
        self.results_pane.update_view(self.view_model)

    def _clear_previous_experiment(self):
        """Clears all UI elements and the ViewModel for a new experiment."""
        self.view_model.clear()
        self.results_pane.clear_all()
        self.setup_pane.clear_algorithms_table()

    def _create_default_param_space(self, model_def: dict) -> dict:
        """Creates a detailed, default parameter space for a given model,
        preserving properties like 'scale' for random sampling.
        """
        param_space = {}
        for param_group in model_def.get("hyperparameter_schema", {}).values():
            for param_name, properties in param_group.items():
                # Pass the whole dictionary of properties to the backend
                param_space[param_name] = properties.copy()
        return param_space

    # --- Event Handlers and Slots ---

    def on_trial_selected(self, trial_id: str):
        """Handles trial selection from the results pane."""
        highlight_ids = {trial_id} if trial_id else set()
        self.results_pane.update_plot_highlight(highlight_ids, self.view_model)

    def on_trial_double_clicked(self, trial_id: str):
        """Handles double-clicking a trial to show its hyperparameters."""
        trial = self.view_model.trials.get(trial_id)
        if trial:
            self.show_hyperparameter_dialog(trial.hyperparameters)

    def on_insight_selected(self, highlight_ids: set):
        """Handles insight selection from the results pane."""
        self.results_pane.update_plot_highlight(highlight_ids, self.view_model)

    def show_hyperparameter_dialog(self, hparams: dict):
        """Shows the hyperparameter viewer dialog for the given parameters."""
        if not hparams:
            QMessageBox.information(self, "Info", "No hyperparameters to display.")
            return
        dialog = HyperparameterViewerDialog(hparams, self)
        dialog.exec()

    def append_log_message(self, log_data: dict):
        self.results_pane.append_log_message(log_data)

    def add_models_to_run(self):
        """Adds newly selected models to an already running experiment."""
        existing_algo_names = {algo.name for algo in self.view_model.algorithms.values()}
        newly_selected_models = self.setup_pane.get_newly_selected_models(existing_algo_names)

        if not newly_selected_models:
            self.append_log_message({'level': 'INFO', 'message': "No new models selected to add."})
            return

        # Prompt the user for the number of trials to generate for the new models
        num_trials, ok = QInputDialog.getInt(
            self,
            "Add Trials",
            f"How many trials to generate for {len(newly_selected_models)} new model(s)?",
            10,  # Default value
            1,   # Minimum value
            1000, # Maximum value
            1,   # Step
        )

        if not ok:
            self.append_log_message({'level': 'INFO', 'message': "Add models operation cancelled by user."})
            return

        self.append_log_message({
            'level': 'INFO',
            'message': f"Adding {num_trials} trials for new models: {', '.join(newly_selected_models)}"
        })
        for model_name in newly_selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = self._create_default_param_space(model_def)
            self.orchestrator.dispatch(
                ActionType.ADD_ALGORITHM,
                {"name": model_name, "parameter_space": param_space, "num_trials": num_trials},
            )

    def remove_algorithm(self, algorithm_id: str):
        """Dispatches an action to remove an algorithm from the experiment."""
        self.orchestrator.dispatch(ActionType.REMOVE_ALGORITHM, {"algorithm_id": algorithm_id})

    def update_throttle(self, value: int):
        """Dispatches an action to update the worker throttle percentage."""
        self.orchestrator.dispatch(ActionType.SET_BUDGET, {"worker_throttle_percent": value})

    def pause_experiment(self):
        """Dispatches an action to pause the current experiment run."""
        self.orchestrator.dispatch(ActionType.PAUSE_RUN, {})

    def resume_experiment(self):
        """Dispatches an action to resume a paused experiment run."""
        self.orchestrator.dispatch(ActionType.RESUME_RUN, {})

    def stop_experiment(self):
        """Dispatches an action to stop the current experiment run."""
        self.orchestrator.dispatch(ActionType.STOP_RUN, {})

    def on_operation_started(self, message: str):
        """Shows the modal progress dialog when a long operation starts."""
        self.progress_dialog.setLabelText(message)
        self.progress_dialog.show()

    def on_operation_finished(self, message: str):
        """Hides the progress dialog when the operation is complete."""
        self.progress_dialog.hide()
        self.append_log_message({"level": "INFO", "message": message})


    def save_experiment(self):
        """Opens a file dialog and dispatches the action to save the experiment."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Experiment", "", "SDE JSON Files (*.sde.json)"
        )
        if filepath:
            self.orchestrator.dispatch(
                ActionType.SAVE_EXPERIMENT, {"filepath": filepath}
            )

    def load_experiment(self):
        """Opens a file dialog and dispatches the action to load an experiment."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Experiment", "", "SDE JSON Files (*.sde.json)"
        )
        if filepath:
            # Clear the UI immediately for a better user experience
            self._clear_previous_experiment()
            self.orchestrator.dispatch(
                ActionType.LOAD_EXPERIMENT, {"filepath": filepath}
            )

    def prune_trial(self, trial_id: str):
        """Dispatches an action to manually prune a trial."""
        trial = self.view_model.trials.get(trial_id)
        if trial:
            trial.status = "PRUNED"
            self.results_pane.update_view(self.view_model)
        self.orchestrator.dispatch(ActionType.MANUAL_PRUNE_TRIAL, {"trial_id": trial_id})

    def prioritize_trial(self, trial_id: str):
        """Dispatches an action to increase a trial's priority."""
        trial = self.view_model.trials.get(trial_id)
        if trial:
            trial.prioritized = True
            self.results_pane.update_view(self.view_model)
        self.orchestrator.dispatch(ActionType.MANUAL_PRIORITIZE_TRIAL, {"trial_id": trial_id})

    def spawn_trial(self, trial_id: str):
        """Opens a dialog to edit hyperparameters and spawn a new trial."""
        source_trial = self.view_model.trials.get(trial_id)
        if not source_trial:
            return

        dialog = SpawnDialog(source_trial.hyperparameters, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_hparams = dialog.get_hyperparameters()
            self.orchestrator.dispatch(
                ActionType.SPAWN_SIMILAR_TRIAL,
                {"source_trial_id": trial_id, "new_hparams": new_hparams},
            )

    def request_state_update(self):
        """Dispatches an action to request a full state update from the orchestrator."""
        self.orchestrator.dispatch(ActionType.REQUEST_STATE_UPDATE, {})

    def closeEvent(self, event):
        """Handles the window close event to ensure graceful shutdown."""
        self.append_log_message({'level': 'INFO', 'message': "Close event received. Shutting down backend engine..."})
        self.orchestrator.shutdown()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
