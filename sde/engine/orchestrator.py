import logging
import traceback
import threading
from typing import List, Callable, Dict, Any

from sde.core.types import Experiment, ExperimentStatus, Trial, TrialStatus
from sde.engine.runtime import SdeRuntimeEngine
from sde.exploration.schedulers import SuccessiveHalvingScheduler, HyperbandScheduler
from sde.challenges import AVAILABLE_DATASETS


logger = logging.getLogger(__name__)


SCHEDULER_MAP = {
    "SuccessiveHalving": SuccessiveHalvingScheduler,
    "Hyperband": HyperbandScheduler,
}

class Signal:
    """A simple signal implementation to remove Qt dependency from the core engine."""
    def __init__(self, *arg_types):
        self._callbacks: List[Callable] = []

    def connect(self, callback: Callable):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.error(f"Error in signal callback: {traceback.format_exc()}")

class ExperimentOrchestrator:
    """
    V2 Orchestrator: Manages the canonical Experiment state object and
    orchestrates the SDE Runtime Engine based on user actions.
    """
    # Signals to update the UI
    state_changed = Signal(dict)
    log_message = Signal(str)

    def __init__(self):
        self.experiment = Experiment()
        self.runtime_engine = None # Will be initialized on START_RUN
        self._lock = threading.Lock()
        self.log_message.emit("INFO: Orchestrator initialized in DEFINING state.")

    def dispatch(self, action_type: str, payload: Dict[str, Any]):
        """
        Receives an action, validates it, mutates the state,
        and triggers side effects.
        """
        with self._lock:
            handler = getattr(self, f"handle_{action_type.lower()}", None)
            if not handler:
                self.log_message.emit(f"ERROR: No handler for action '{action_type}'")
                return

            # Use the new structured validator
            valid_actions = self.get_valid_actions()
            if not self._is_action_valid(action_type, payload, valid_actions):
                self.log_message.emit(f"WARN: Action '{action_type}' is not valid for the current state or payload.")
                return

            try:
                handler(payload)
                self.emit_state_change()
            except Exception as e:
                self.log_message.emit(f"ERROR: Failed to execute action {action_type}: {e}")
                logger.error(traceback.format_exc())

    def _is_action_valid(self, action_type: str, payload: Dict[str, Any], valid_actions: Dict) -> bool:
        """Checks if a given action is present in the structured valid_actions dict."""
        if action_type in valid_actions.get('global', []):
            return True

        if 'algorithm_id' in payload:
            algo_id = payload['algorithm_id']
            if action_type in valid_actions.get('algorithms', {}).get(algo_id, []):
                return True

        if 'trial_id' in payload:
            trial_id = payload['trial_id']
            if action_type in valid_actions.get('trials', {}).get(trial_id, []):
                return True

        # Fallback for actions that don't have a specific context object
        # This is less specific but maintains backwards compatibility for simple checks.
        all_actions = set(valid_actions.get('global', []))
        for aname, alist in valid_actions.get('algorithms', {}).items():
            all_actions.update(alist)
        for tname, tlist in valid_actions.get('trials', {}).items():
            all_actions.update(tlist)

        return action_type in all_actions


    def emit_state_change(self):
        """Serializes the experiment state and emits it."""
        # A real implementation would have a more robust serializer
        state_dict = {
            "id": self.experiment.id,
            "status": self.experiment.status.value,
            "challenge": self.experiment.challenge,
            "algorithms": {k: v.__dict__ for k, v in self.experiment.algorithms.items()},
            "trials": {k: v.to_dict() for k, v in self.experiment.trials.items()},
            "insights": self.experiment.insights,
            "adaptive_policy": self.experiment.adaptive_policy,
            "valid_actions": self.get_valid_actions(), # Also emit the valid actions
        }
        self.state_changed.emit(state_dict)

    # --- Action Handlers ---

    def handle_set_challenge(self, payload: Dict[str, Any]):
        self.experiment.challenge = payload
        self.log_message.emit(f"INFO: Challenge set to '{payload.get('name', 'Unknown')}'")

    def handle_add_algorithm(self, payload: Dict[str, Any]):
        # In a real app, this would do more validation
        from sde.core.types import AlgorithmConfig
        algo_id = f"algo_{len(self.experiment.algorithms)}"
        new_algo = AlgorithmConfig(
            id=algo_id,
            name=payload['name'],
            parameter_space=payload['parameter_space']
        )
        self.experiment.algorithms[algo_id] = new_algo
        self.log_message.emit(f"INFO: Added algorithm: {new_algo.name}")

    def handle_remove_algorithm(self, payload: Dict[str, Any]):
        algo_id = payload['algorithm_id']
        if algo_id in self.experiment.algorithms:
            algo_name = self.experiment.algorithms[algo_id].name
            del self.experiment.algorithms[algo_id]

            # Prune associated trials
            for trial in self.experiment.trials.values():
                if trial.algorithm_name == algo_name:
                    trial.status = TrialStatus.PRUNED
                    if self.runtime_engine:
                        self.runtime_engine.cancel_work_for_trial(trial.id)

            self.log_message.emit(f"INFO: Removed algorithm '{algo_name}' and pruned its trials.")
        else:
            self.log_message.emit(f"WARN: Could not find algorithm with id {algo_id} to remove.")

    def handle_update_param_space(self, payload: Dict[str, Any]):
        algo_id = payload['algorithm_id']
        new_space = payload['new_space']
        if algo_id in self.experiment.algorithms:
            self.experiment.algorithms[algo_id].parameter_space = new_space
            self.log_message.emit(f"INFO: Updated parameter space for algorithm {algo_id}.")
        else:
            self.log_message.emit(f"WARN: Could not find algorithm with id {algo_id} to update.")

    def handle_set_adaptive_policy(self, payload: Dict[str, Any]):
        policy_name = payload['policy_name']
        if policy_name in SCHEDULER_MAP:
            self.experiment.adaptive_policy = policy_name
            self.log_message.emit(f"INFO: Adaptive policy set to '{policy_name}'.")
        else:
            self.log_message.emit(f"ERROR: Unknown policy name '{policy_name}'.")

    def handle_set_budget(self, payload: Dict[str, Any]):
        self.experiment.patience_budget = payload
        self.log_message.emit(f"INFO: Patience budget set to {payload}.")

    def handle_manual_prune_trial(self, payload: Dict[str, Any]):
        trial_id = payload['trial_id']
        if trial_id in self.experiment.trials:
            self.experiment.trials[trial_id].status = TrialStatus.PRUNED
            if self.runtime_engine:
                self.runtime_engine.cancel_work_for_trial(trial_id)
            self.log_message.emit(f"INFO: Manually pruned trial {trial_id}.")
        else:
            self.log_message.emit(f"WARN: Could not find trial with id {trial_id} to prune.")

    def handle_manual_prioritize_trial(self, payload: Dict[str, Any]):
        trial_id = payload['trial_id']
        if trial_id in self.experiment.trials:
            # Increase priority by a fixed amount
            self.experiment.trials[trial_id].priority += 10
            self.log_message.emit(f"INFO: Increased priority for trial {trial_id}.")
        else:
            self.log_message.emit(f"WARN: Could not find trial with id {trial_id} to prioritize.")

    def handle_spawn_similar_trial(self, payload: Dict[str, Any]):
        source_trial_id = payload['source_trial_id']
        if source_trial_id not in self.experiment.trials:
            self.log_message.emit(f"WARN: Could not find source trial {source_trial_id} to spawn from.")
            return

        source_trial = self.experiment.trials[source_trial_id]
        new_hparams = payload.get('new_hparams', source_trial.hyperparameters.copy())

        new_trial_id = f"trial_{source_trial.algorithm_name.lower()}_{len(self.experiment.trials)}"
        new_trial = Trial(
            id=new_trial_id,
            algorithm_name=source_trial.algorithm_name,
            hyperparameters=new_hparams,
            status=TrialStatus.PENDING
        )
        self.experiment.trials[new_trial_id] = new_trial

        if self.runtime_engine:
            self.runtime_engine.add_trial_live(new_trial)

        self.log_message.emit(f"INFO: Spawned new trial {new_trial_id} from {source_trial_id}.")

    def handle_start_run(self, payload: Dict[str, Any]):
        self.log_message.emit("INFO: START_RUN action received. Validating and initializing runtime.")
        if not self.experiment.algorithms:
            self.log_message.emit("ERROR: Cannot start run without at least one algorithm.")
            return

        self.experiment.status = ExperimentStatus.RUNNING

        # --- V2 Trial Generation ---
        if not self.experiment.trials:
            self.log_message.emit("INFO: No pre-existing trials found. Generating initial trials.")
            import random
            import numpy as np

            num_trials_per_algo = 10 # Default number of trials for random search

            for algo_config in self.experiment.algorithms.values():
                for i in range(num_trials_per_algo):
                    hparams = {}
                    for p_name, p_def in algo_config.parameter_space.items():
                        # This logic is inspired by the UI's random search generation
                        # A more robust implementation would use a schema
                        if isinstance(p_def, dict) and 'min' in p_def and 'max' in p_def:
                            if p_def.get('scale') == 'log':
                                log_min = np.log10(p_def['min'])
                                log_max = np.log10(p_def['max'])
                                value = 10**random.uniform(log_min, log_max)
                            else:
                                value = random.uniform(p_def['min'], p_def['max'])

                            if p_def.get('type') == 'int':
                                value = int(value)
                        elif isinstance(p_def, (list, tuple)): # Simple range tuple or list of choices
                             if all(isinstance(x, (int, float)) for x in p_def) and len(p_def) == 2:
                                 value = random.uniform(p_def[0], p_def[1]) # Assume (min, max)
                             else:
                                 value = random.choice(p_def) # Assume list of choices
                        else:
                            value = p_def # A fixed value
                        hparams[p_name] = value

                    trial_id = f"trial_{algo_config.name.lower().replace(' ', '_')}_{len(self.experiment.trials)}"
                    trial = Trial(
                        id=trial_id,
                        algorithm_name=algo_config.name,
                        hyperparameters=hparams,
                    )
                    self.experiment.trials[trial_id] = trial
            self.log_message.emit(f"INFO: Generated {len(self.experiment.trials)} initial trials via random search.")

        # --- Engine Initialization ---
        self._initialize_and_start_runtime()

    def _initialize_and_start_runtime(self):
        """Creates and starts a new SdeRuntimeEngine instance."""
        try:
            challenge_name = self.experiment.challenge['name']
            challenge_def = AVAILABLE_DATASETS[challenge_name]

            scheduler_name = self.experiment.adaptive_policy
            scheduler_class = SCHEDULER_MAP.get(scheduler_name)
            if not scheduler_class:
                raise ValueError(f"Unknown scheduler '{scheduler_name}' specified in adaptive_policy.")

            increasing = "accuracy" in challenge_def.performance_metric_name.lower()
            scheduler = scheduler_class(metric=challenge_def.performance_metric_name, increasing=increasing)

            # Important: Pass a copy of the list of trials to the engine
            current_trials = list(self.experiment.trials.values())

            self.runtime_engine = SdeRuntimeEngine(
                trials=current_trials,
                dataset_name=challenge_name,
                adaptive_scheduler=scheduler,
                trial_updated_callback=self.on_trial_updated,
                insights_callback=self.on_insights_generated
            )

            self.log_message.emit(f"INFO: SdeRuntimeEngine initialized with {scheduler_name} scheduler.")
            self.runtime_engine.start()
            self.log_message.emit("INFO: SdeRuntimeEngine started successfully.")

        except Exception as e:
            self.log_message.emit(f"ERROR: Failed to start runtime engine: {e}")
            logger.error(f"Engine start failed: {traceback.format_exc()}")
            self.experiment.status = ExperimentStatus.DEFINING # Revert status on failure

    def on_trial_updated(self, trial_data: Dict[str, Any]):
        """
        Callback for the SdeRuntimeEngine to update the orchestrator's state.
        This method is called from the engine's thread.
        """
        with self._lock:
            trial_id = trial_data.get('id')
            if not trial_id or trial_id not in self.experiment.trials:
                logger.warning(f"Orchestrator received update for unknown trial_id: {trial_id}")
                return

            # Update the trial object in our central state
            trial = self.experiment.trials[trial_id]
            trial.status = TrialStatus(trial_data['status'])
            trial.current_epoch = trial_data['current_epoch']
            trial.est_time_per_epoch = trial_data['est_time_per_epoch']
            trial.results = trial_data['results']
            # Note: A more robust implementation might use a proper deserializer
            # that reconstructs the Trial object fully.

        # Emit state change to notify UI
        self.emit_state_change()

    def on_insights_generated(self, insights: List[Dict[str, Any]]):
        """
        Callback for the SdeRuntimeEngine to add new insights to the state.
        """
        with self._lock:
            self.experiment.insights.extend(insights)
            for insight in insights:
                self.log_message.emit(f"INSIGHT: {insight['message']}")
        self.emit_state_change()

    def handle_pause_run(self, payload: Dict[str, Any]):
        if self.runtime_engine:
            self.runtime_engine.stop()
            self.experiment.status = ExperimentStatus.PAUSED
            self.log_message.emit("INFO: Experiment paused.")

    def handle_resume_run(self, payload: Dict[str, Any]):
        self.log_message.emit("INFO: RESUME_RUN action received. Re-initializing runtime.")
        self.experiment.status = ExperimentStatus.RUNNING
        # Re-initialize the engine with the current state of trials
        self._initialize_and_start_runtime()

    def get_valid_actions(self) -> Dict[str, Any]:
        """
        This is the Action Validator. It inspects the current state and returns
        a structured dictionary of valid actions, separated by context.
        e.g., {
            "global": ["ADD_ALGORITHM"],
            "algorithms": { "algo_1": ["UPDATE_PARAM_SPACE", "REMOVE_ALGORITHM"] },
            "trials": { "trial_abc": ["MANUAL_PRUNE_TRIAL"] }
        }
        """
        status = self.experiment.status
        actions: Dict[str, Any] = {
            "global": [],
            "algorithms": {},
            "trials": {},
        }
        has_challenge = self.experiment.challenge is not None

        # Global actions
        if status == ExperimentStatus.DEFINING:
            if not has_challenge:
                actions["global"].append("SET_CHALLENGE")
            else:
                actions["global"].append("ADD_ALGORITHM")
                actions["global"].append("SET_ADAPTIVE_POLICY")
                actions["global"].append("SET_BUDGET")
                if self.experiment.algorithms:
                    actions["global"].append("START_RUN")

        elif status == ExperimentStatus.RUNNING:
            actions["global"].append("PAUSE_RUN")
            actions["global"].append("ADD_ALGORITHM") # Can always add new contenders

        elif status == ExperimentStatus.PAUSED:
            actions["global"].append("RESUME_RUN")
            actions["global"].append("ADD_ALGORITHM")

        # Per-algorithm actions
        for algo_id, algo in self.experiment.algorithms.items():
            algo_actions = []
            if status == ExperimentStatus.DEFINING:
                algo_actions.append("UPDATE_PARAM_SPACE")
                algo_actions.append("REMOVE_ALGORITHM")
            elif status == ExperimentStatus.RUNNING or status == ExperimentStatus.PAUSED:
                 algo_actions.append("REMOVE_ALGORITHM")

            if algo_actions:
                actions["algorithms"][algo_id] = algo_actions

        # Per-trial actions
        for trial_id, trial in self.experiment.trials.items():
            trial_actions = []
            # Can spawn from any trial that has finished at least one step
            if trial.results:
                 trial_actions.append("SPAWN_SIMILAR_TRIAL")

            if trial.status == TrialStatus.ACTIVE:
                trial_actions.append("MANUAL_PRUNE_TRIAL")
                trial_actions.append("MANUAL_PRIORITIZE_TRIAL")

            if trial_actions:
                actions["trials"][trial_id] = trial_actions

        return actions
