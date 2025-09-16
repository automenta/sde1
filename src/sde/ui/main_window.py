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
from .dialog_service import DialogService
from .view_controller import ViewController
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
        self.view_controller = ViewController(self.orchestrator)
        self.view_model = ExperimentViewModel(self.style())
        self.dialog_service = DialogService(self)

        self._init_ui()
        self._connect_signals()

        self.append_log_message(
            {
                "level": "INFO",
                "message": "UI Initialized. Configure your experiment and click 'Start'.",
            }
        )

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
        self.progress_dialog.setMaximum(0)  # Makes it an indeterminate progress bar

    def _connect_signals(self):
        """Connects all UI signals to their corresponding slots."""
        # Backend signals -> Main Window
        self.view_controller.log_message.connect(self.append_log_message)
        self.view_controller.state_changed.connect(self.on_state_changed)
        self.view_controller.operation_started.connect(self.on_operation_started)
        self.view_controller.operation_finished.connect(self.on_operation_finished)

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
            self.append_log_message(
                {"level": "ERROR", "message": f"Unknown run mode: {run_mode}"}
            )

    def start_simple_experiment(self, settings: dict):
        """Handles the 'Simple' run mode by showing the SimpleRunDialog."""
        dataset_name, selected_models_names = self.setup_pane.get_experiment_settings()
        if not dataset_name or not selected_models_names:
            self.dialog_service.show_warning(
                "Missing Information",
                "Please select a dataset and at least one model.",
            )
            return

        result, custom_hparams = self.dialog_service.show_simple_run_dialog(
            selected_models_names
        )

        if result is None:
            self.append_log_message(
                {"level": "INFO", "message": "Experiment start cancelled by user."}
            )
            return

        if custom_hparams:
            self.append_log_message(
                {
                    "level": "INFO",
                    "message": "Starting run with custom hyperparameters.",
                }
            )
        else:
            self.append_log_message(
                {
                    "level": "INFO",
                    "message": "Starting run with default hyperparameters.",
                }
            )

        self.append_log_message(
            {"level": "INFO", "message": f"Configuring experiment on '{dataset_name}'"}
        )
        self._clear_previous_experiment()
        self.view_controller.start_simple_experiment(
            settings, dataset_name, selected_models_names, custom_hparams
        )

    def open_tuning_dialog(self, settings: dict):
        """Opens the tuning dialog and configures the experiment via the orchestrator."""
        dataset_name, selected_models_names = self.setup_pane.get_experiment_settings()
        if not dataset_name or not selected_models_names:
            self.dialog_service.show_warning(
                "Warning",
                "Please select a dataset and at least one model to tune.",
            )
            return

        config = self.dialog_service.show_tuning_dialog(selected_models_names)
        if not config:
            return

        self.append_log_message(
            {"level": "INFO", "message": f"Configuring experiment on '{dataset_name}'"}
        )
        self._clear_previous_experiment()

        self.view_controller.start_tuning_experiment(settings, dataset_name, config)

    # --- UI Update and State Management ---

    def _update_all_widgets(self):
        """Refreshes all UI components based on the current ViewModel state."""
        self.setup_pane.update_button_states(self.view_model)
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
        self.results_pane.update_plot_highlight(highlight_ids)

    def on_trial_double_clicked(self, trial_id: str):
        """Handles double-clicking a trial to show its hyperparameters."""
        trial = self.view_model.trials.get(trial_id)
        if trial:
            self.dialog_service.show_hyperparameter_viewer(trial.hyperparameters)

    def on_insight_selected(self, highlight_ids: set):
        """Handles insight selection from the results pane."""
        self.results_pane.update_plot_highlight(highlight_ids)

    def show_hyperparameter_dialog(self, hparams: dict):
        """Shows the hyperparameter viewer dialog for the given parameters."""
        self.dialog_service.show_hyperparameter_viewer(hparams)

    def append_log_message(self, log_data: dict):
        self.results_pane.append_log_message(log_data)

    def add_models_to_run(self):
        """Adds newly selected models to an already running experiment."""
        existing_algo_names = {
            algo.name for algo in self.view_model.algorithms.values()
        }
        newly_selected_models = self.setup_pane.get_newly_selected_models(
            existing_algo_names
        )

        if not newly_selected_models:
            self.append_log_message(
                {"level": "INFO", "message": "No new models selected to add."}
            )
            return

        num_trials = self.dialog_service.show_add_trials_dialog(
            len(newly_selected_models)
        )

        if not num_trials:
            self.append_log_message(
                {"level": "INFO", "message": "Add models operation cancelled by user."}
            )
            return

        self.append_log_message(
            {
                "level": "INFO",
                "message": f"Adding {num_trials} trials for new models: {', '.join(newly_selected_models)}",
            }
        )
        for model_name in newly_selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = self._create_default_param_space(model_def)
            self.view_controller.add_models_to_run(
                model_name, param_space, num_trials
            )

    def remove_algorithm(self, algorithm_id: str):
        """Dispatches an action to remove an algorithm from the experiment."""
        self.view_controller.remove_algorithm(algorithm_id)

    def update_throttle(self, value: int):
        """Dispatches an action to update the worker throttle percentage."""
        self.view_controller.update_throttle(value)

    def pause_experiment(self):
        """Dispatches an action to pause the current experiment run."""
        self.view_controller.pause_experiment()

    def resume_experiment(self):
        """Dispatches an action to resume a paused experiment run."""
        self.view_controller.resume_experiment()

    def stop_experiment(self):
        """Dispatches an action to stop the current experiment run."""
        self.view_controller.stop_experiment()

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
        filepath = self.dialog_service.show_save_experiment_dialog()
        if filepath:
            self.view_controller.save_experiment(filepath)

    def load_experiment(self):
        """Opens a file dialog and dispatches the action to load an experiment."""
        filepath = self.dialog_service.show_load_experiment_dialog()
        if filepath:
            # Clear the UI immediately for a better user experience
            self._clear_previous_experiment()
            self.view_controller.load_experiment(filepath)

    def prune_trial(self, trial_id: str):
        """Dispatches an action to manually prune a trial."""
        self.view_controller.prune_trial(trial_id)

    def prioritize_trial(self, trial_id: str):
        """Dispatches an action to increase a trial's priority."""
        self.view_controller.prioritize_trial(trial_id)

    def spawn_trial(self, trial_id: str):
        """Opens a dialog to edit hyperparameters and spawn a new trial."""
        source_trial = self.view_model.trials.get(trial_id)
        if not source_trial:
            return

        new_hparams = self.dialog_service.show_spawn_dialog(source_trial.hyperparameters)
        if new_hparams:
            self.view_controller.spawn_trial(trial_id, new_hparams)

    def request_state_update(self):
        """Dispatches an action to request a full state update from the orchestrator."""
        self.view_controller.request_state_update()

    def closeEvent(self, event):
        """Handles the window close event to ensure graceful shutdown."""
        self.append_log_message(
            {
                "level": "INFO",
                "message": "Close event received. Shutting down backend engine...",
            }
        )
        self.view_controller.shutdown()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
