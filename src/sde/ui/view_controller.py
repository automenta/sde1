from PyQt6.QtCore import QObject, pyqtSignal
from ..core.actions import ActionType
from ..engine.orchestrator import ExperimentOrchestrator
from ..models import AVAILABLE_MODELS
from ..challenges import AVAILABLE_DATASETS

class ViewController(QObject):
    log_message = pyqtSignal(dict)
    state_changed = pyqtSignal(dict)
    operation_started = pyqtSignal(str)
    operation_finished = pyqtSignal(str)

    def __init__(self, orchestrator: ExperimentOrchestrator):
        super().__init__()
        self.orchestrator = orchestrator
        self._connect_orchestrator_signals()

    def _connect_orchestrator_signals(self):
        self.orchestrator.log_message.connect(self.log_message)
        self.orchestrator.state_changed.connect(self.state_changed)
        self.orchestrator.operation_started.connect(self.operation_started)
        self.orchestrator.operation_finished.connect(self.operation_finished)

    def dispatch(self, action_type, payload):
        self.orchestrator.dispatch(action_type, payload)

    def start_simple_experiment(self, settings, dataset_name, selected_models_names, custom_hparams):
        algorithms_to_add = []
        for model_name in selected_models_names:
            model_def = AVAILABLE_MODELS[model_name]
            param_space = {}
            if custom_hparams and model_name in custom_hparams:
                param_space = custom_hparams[model_name]
            else:
                for param_type, params in model_def.hyperparameter_schema.items():
                    for param_name, properties in params.items():
                        param_space[param_name] = properties.get("default")
            algorithms_to_add.append(
                {
                    "name": model_name,
                    "parameter_space": param_space,
                    "is_simple_run": True,
                }
            )
        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.start_experiment(settings, dataset_name, algorithms_to_add, challenge_def)

    def start_tuning_experiment(self, settings, dataset_name, config):
        self.dispatch(
            ActionType.SET_ADAPTIVE_POLICY,
            {"policy_name": config["adaptive_scheduler"]},
        )

        algorithms_to_add = []
        for model_name, model_params in config["models"].items():
            full_param_space = {}
            for param_type in model_params.values():
                for param_name, properties in param_type.items():
                    full_param_space[param_name] = {
                        "type": "float",
                        "min": properties["min"],
                        "max": properties["max"],
                        "scale": properties.get("scale", "linear"),
                    }
            algorithms_to_add.append(
                {"name": model_name, "parameter_space": full_param_space}
            )
        challenge_def = AVAILABLE_DATASETS[dataset_name]
        self.start_experiment(settings, dataset_name, algorithms_to_add, challenge_def)

    def start_experiment(self, settings, dataset_name, algorithms_to_add, challenge_def):
        self.dispatch(
            ActionType.SET_CHALLENGE,
            {"name": dataset_name, "type": challenge_def.type.value},
        )

        for algo_config in algorithms_to_add:
            self.dispatch(ActionType.ADD_ALGORITHM, algo_config)

        self.dispatch(ActionType.START_RUN, settings)

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
