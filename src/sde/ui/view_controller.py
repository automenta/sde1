from PyQt6.QtCore import QObject, pyqtSignal

from ..challenges import AVAILABLE_DATASETS
from ..core.actions import ActionType
from ..engine.orchestrator import ExperimentOrchestrator
from ..models import AVAILABLE_MODELS


class ViewController(QObject):
    """Handles user interactions and orchestrates calls to the backend."""

    log_message = pyqtSignal(dict)
    state_changed = pyqtSignal(dict)
    operation_started = pyqtSignal(str)
    operation_finished = pyqtSignal(str)

    def __init__(self, orchestrator: ExperimentOrchestrator, view_model, dialog_service):
        super().__init__()
        self.orchestrator = orchestrator
        self.view_model = view_model
        self.dialog_service = dialog_service
        self._connect_orchestrator_signals()

    def _connect_orchestrator_signals(self):
        self.orchestrator.log_message.connect(self.log_message)
        self.orchestrator.state_changed.connect(self.state_changed)
        self.orchestrator.operation_started.connect(self.operation_started)
        self.orchestrator.operation_finished.connect(self.operation_finished)

    def dispatch(self, action_type, payload):
        self.orchestrator.dispatch(action_type, payload)

    # --- Experiment Lifecycle ---

    def initiate_experiment_start(
        self, settings, dataset_name, selected_models_names
    ):
        """Orchestrates the process of starting a new experiment."""
        run_mode = settings.get("run_mode")
        if run_mode == "Simple":
            self._initiate_simple_experiment(
                settings, dataset_name, selected_models_names
            )
        elif run_mode == "Tune Hyperparameters":
            self._initiate_tuning_experiment(
                settings, dataset_name, selected_models_names
            )
        else:
            self.log_message.emit(
                {"level": "ERROR", "message": f"Unknown run mode: {run_mode}"}
            )

    def _initiate_simple_experiment(
        self, settings, dataset_name, selected_models_names
    ):
        result, custom_hparams = self.dialog_service.show_simple_run_dialog(
            selected_models_names
        )
        if result is None:
            self.log_message.emit(
                {"level": "INFO", "message": "Experiment start cancelled by user."}
            )
            return

        self.log_message.emit(
            {
                "level": "INFO",
                "message": "Starting run with {} hyperparameters.".format(
                    "custom" if custom_hparams else "default"
                ),
            }
        )

        algorithms_to_add = []
        for model_name in selected_models_names:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = {}
            if custom_hparams and model_name in custom_hparams:
                param_space = custom_hparams[model_name]
            else:
                # Create default param space from schema
                for params in model_def.get("hyperparameter_schema", {}).values():
                    for param_name, properties in params.items():
                        param_space[param_name] = properties.get("default")
            algorithms_to_add.append({"name": model_name, "parameter_space": param_space})

        self._dispatch_experiment_start(settings, dataset_name, algorithms_to_add)

    def _initiate_tuning_experiment(
        self, settings, dataset_name, selected_models_names
    ):
        config = self.dialog_service.show_tuning_dialog(selected_models_names)
        if not config:
            return  # User cancelled

        self.dispatch(
            ActionType.SET_ADAPTIVE_POLICY,
            {"policy_name": config["adaptive_scheduler"]},
        )

        algorithms_to_add = []
        for model_name, model_params in config["models"].items():
            param_space = {}
            for param_type in model_params.values():
                for param_name, properties in param_type.items():
                    param_space[param_name] = {
                        "type": "float",
                        "min": properties["min"],
                        "max": properties["max"],
                        "scale": properties.get("scale", "linear"),
                    }
            algorithms_to_add.append({"name": model_name, "parameter_space": param_space})

        self._dispatch_experiment_start(settings, dataset_name, algorithms_to_add)

    def _dispatch_experiment_start(
        self, settings, dataset_name, algorithms_to_add
    ):
        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.dispatch(
            ActionType.SET_CHALLENGE,
            {"name": dataset_name, "type": challenge_def.type.value},
        )
        for algo_config in algorithms_to_add:
            self.dispatch(ActionType.ADD_ALGORITHM, algo_config)

        self.dispatch(ActionType.START_RUN, settings)

    def initiate_add_models_to_run(self, newly_selected_models):
        """Handles the logic for adding new models to a running experiment."""
        if not newly_selected_models:
            self.log_message.emit(
                {"level": "INFO", "message": "No new models selected to add."}
            )
            return

        num_trials = self.dialog_service.show_add_trials_dialog(
            len(newly_selected_models)
        )

        if not num_trials:
            self.log_message.emit(
                {"level": "INFO", "message": "Add models operation cancelled by user."}
            )
            return

        self.log_message.emit(
            {
                "level": "INFO",
                "message": f"Adding {num_trials} trials for new models: {', '.join(newly_selected_models)}",
            }
        )
        for model_name in newly_selected_models:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = self._create_default_param_space(model_def)
            self.add_models_to_run(model_name, param_space, num_trials)

    # --- UI Interaction Handlers ---

    def show_trial_hyperparameters(self, trial_id: str):
        """Shows the hyperparameters for a given trial."""
        trial = self.view_model.trials.get(trial_id)
        if trial:
            self.dialog_service.show_hyperparameter_viewer(trial.hyperparameters)

    def initiate_spawn_trial(self, trial_id: str):
        """Handles the UI flow for spawning a new trial from an existing one."""
        source_trial = self.view_model.trials.get(trial_id)
        if not source_trial:
            return

        new_hparams = self.dialog_service.show_spawn_dialog(
            source_trial.hyperparameters
        )
        if new_hparams:
            self.spawn_trial(trial_id, new_hparams)

    def _create_default_param_space(self, model_def: dict) -> dict:
        """Creates a detailed, default parameter space for a given model."""
        param_space = {}
        for param_group in model_def.get("hyperparameter_schema", {}).values():
            for param_name, properties in param_group.items():
                param_space[param_name] = properties.copy()
        return param_space

    # --- Direct Action Dispatchers ---

    def pause_experiment(self):
        self.dispatch(ActionType.PAUSE_RUN, {})

    def resume_experiment(self):
        self.dispatch(ActionType.RESUME_RUN, {})

    def stop_experiment(self):
        self.dispatch(ActionType.STOP_RUN, {})

    def save_experiment(self, filepath):
        self.dispatch(ActionType.SAVE_EXPERIMENT, {"filepath": filepath})

    def load_experiment(self, filepath):
        self.dispatch(ActionType.LOAD_EXPERIMENT, {"filepath": filepath})

    def add_models_to_run(self, model_name, param_space, num_trials):
        self.dispatch(
            ActionType.ADD_ALGORITHM,
            {
                "name": model_name,
                "parameter_space": param_space,
                "num_trials": num_trials,
            },
        )

    def remove_algorithm(self, algorithm_id):
        self.dispatch(ActionType.REMOVE_ALGORITHM, {"algorithm_id": algorithm_id})

    def update_throttle(self, value):
        self.dispatch(ActionType.SET_BUDGET, {"worker_throttle_percent": value})

    def prune_trial(self, trial_id):
        self.dispatch(ActionType.MANUAL_PRUNE_TRIAL, {"trial_id": trial_id})

    def prioritize_trial(self, trial_id):
        self.dispatch(ActionType.MANUAL_PRIORITIZE_TRIAL, {"trial_id": trial_id})

    def spawn_trial(self, source_trial_id, new_hparams):
        self.dispatch(
            ActionType.SPAWN_SIMILAR_TRIAL,
            {"source_trial_id": source_trial_id, "new_hparams": new_hparams},
        )

    def request_state_update(self):
        self.dispatch(ActionType.REQUEST_STATE_UPDATE, {})

    def shutdown(self):
        self.orchestrator.shutdown()
