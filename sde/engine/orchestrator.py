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
        from sde.core.types import AlgorithmConfig
        algo_id = f"algo_{len(self.experiment.algorithms)}"
        new_algo = AlgorithmConfig(
            id=algo_id,
            name=payload['name'],
            parameter_space=payload['parameter_space']
        )
        self.experiment.algorithms[algo_id] = new_algo
        self.log_message.emit(f"INFO: Added algorithm: {new_algo.name}")

        # If the experiment is already running, generate trials and add them live
        if self.experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.log_message.emit(f"INFO: Generating new trials for algorithm '{new_algo.name}' mid-run.")
            try:
                scheduler_name = self.experiment.adaptive_policy
                scheduler_class = SCHEDULER_MAP.get(scheduler_name)
                if not scheduler_class:
                    raise ValueError(f"Unknown scheduler '{scheduler_name}' specified in adaptive_policy.")

                challenge_def = AVAILABLE_DATASETS[self.experiment.challenge['name']]
                increasing = "accuracy" in challenge_def.performance_metric_name.lower()
                scheduler = scheduler_class(metric=challenge_def.performance_metric_name, increasing=increasing)

                num_trials_per_algo = 10 # This could be part of the budget definition later
                new_trials = scheduler.generate_initial_trials([new_algo], num_trials_per_algo)

                for trial in new_trials:
                    self.experiment.trials[trial.id] = trial

                if self.runtime_engine:
                    self.runtime_engine.add_trials_live(new_trials)

                self.log_message.emit(f"INFO: Added {len(new_trials)} new trials to the running experiment.")

            except Exception as e:
                self.log_message.emit(f"ERROR: Failed to add new trials mid-run: {e}")
                logger.error(f"Mid-run trial generation failed: {traceback.format_exc()}")

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
        if algo_id not in self.experiment.algorithms:
            self.log_message.emit(f"WARN: Could not find algorithm with id {algo_id} to update.")
            return

        algo = self.experiment.algorithms[algo_id]
        algo.parameter_space = new_space
        self.log_message.emit(f"INFO: Updated parameter space for algorithm {algo.name} ({algo_id}).")

        # If the experiment is running, generate new trials based on the updated space
        if self.runtime_engine and self.experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.log_message.emit(f"INFO: Generating new trials for '{algo.name}' due to parameter space update.")
            try:
                # Use the existing scheduler from the runtime engine
                scheduler = self.runtime_engine.adaptive_scheduler
                if not scheduler:
                     raise ValueError("Runtime engine has no adaptive scheduler available.")

                # Ask the scheduler to generate new trials for just this algorithm
                num_new_trials = 10 # This could be a configurable setting
                new_trials = scheduler.generate_initial_trials([algo], num_new_trials)

                for trial in new_trials:
                    self.experiment.trials[trial.id] = trial

                self.runtime_engine.add_trials_live(new_trials)
                self.log_message.emit(f"INFO: Added {len(new_trials)} new trials to the running experiment for '{algo.name}'.")

            except Exception as e:
                self.log_message.emit(f"ERROR: Failed to generate new trials after param space update: {e}")
                logger.error(f"Failed to generate new trials after param space update: {traceback.format_exc()}")

    def handle_set_adaptive_policy(self, payload: Dict[str, Any]):
        policy_name = payload['policy_name']
        if policy_name not in SCHEDULER_MAP:
            self.log_message.emit(f"ERROR: Unknown policy name '{policy_name}'.")
            return

        self.experiment.adaptive_policy = policy_name
        self.log_message.emit(f"INFO: Adaptive policy set to '{policy_name}'.")

        # If the run is live, hot-swap the scheduler in the runtime engine
        if self.runtime_engine and self.experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.log_message.emit("INFO: Hot-swapping adaptive policy in live runtime engine.")
            try:
                challenge_def = AVAILABLE_DATASETS[self.experiment.challenge['name']]
                increasing = "accuracy" in challenge_def.performance_metric_name.lower()
                new_scheduler_class = SCHEDULER_MAP[policy_name]

                # --- Instantiate the new scheduler with correct parameters ---
                scheduler_args = {
                    "metric": challenge_def.performance_metric_name,
                    "increasing": increasing,
                }
                if policy_name == "Hyperband":
                    # Hyperband requires max_resource_per_trial. Let's use a default or get from budget.
                    # This part of the design could be improved with a more structured budget.
                    max_resource = self.experiment.patience_budget.get('max_epochs', 81) if self.experiment.patience_budget else 81
                    scheduler_args['max_resource_per_trial'] = max_resource

                new_scheduler = new_scheduler_class(**scheduler_args)
                # --- End of instantiation ---

                self.runtime_engine.update_adaptive_policy(new_scheduler)
                self.log_message.emit("INFO: Adaptive policy updated successfully in runtime.")
            except Exception as e:
                self.log_message.emit(f"ERROR: Failed to hot-swap adaptive policy: {e}")
                logger.error(f"Failed to hot-swap adaptive policy: {traceback.format_exc()}")

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
            # Use the new, more general method for adding trials live
            self.runtime_engine.add_trials_live([new_trial])

        self.log_message.emit(f"INFO: Spawned new trial {new_trial_id} from {source_trial_id}.")

    def handle_start_run(self, payload: Dict[str, Any]):
        self.log_message.emit("INFO: START_RUN action received. Validating and initializing runtime.")
        if not self.experiment.algorithms:
            self.log_message.emit("ERROR: Cannot start run without at least one algorithm.")
            return

        self.experiment.status = ExperimentStatus.RUNNING

        # --- V2 Trial Generation (Delegated) ---
        if not self.experiment.trials:
            self.log_message.emit("INFO: No pre-existing trials found. Delegating to adaptive policy for generation.")
            try:
                # 1. Get info needed to instantiate the scheduler
                scheduler_name = self.experiment.adaptive_policy
                scheduler_class = SCHEDULER_MAP.get(scheduler_name)
                if not scheduler_class:
                    raise ValueError(f"Unknown scheduler '{scheduler_name}' specified in adaptive_policy.")

                challenge_def = AVAILABLE_DATASETS[self.experiment.challenge['name']]
                increasing = "accuracy" in challenge_def.performance_metric_name.lower()

                # A bit of a hack: some schedulers need more params. This should be improved
                # with a better config system. For now, we only pass what's needed for the base case.
                scheduler = scheduler_class(metric=challenge_def.performance_metric_name, increasing=increasing)

                # 2. Ask the scheduler to generate trials
                num_trials_per_algo = 10 # This could be part of the budget definition later
                algorithms = list(self.experiment.algorithms.values())
                new_trials = scheduler.generate_initial_trials(algorithms, num_trials_per_algo)

                # 3. Update the experiment state with the new trials
                for trial in new_trials:
                    self.experiment.trials[trial.id] = trial

                self.log_message.emit(f"INFO: Generated {len(self.experiment.trials)} initial trials via '{scheduler_name}' policy.")

            except Exception as e:
                self.log_message.emit(f"ERROR: Failed to generate initial trials: {e}")
                logger.error(f"Trial generation failed: {traceback.format_exc()}")
                self.experiment.status = ExperimentStatus.DEFINING # Revert status
                return

        # --- Engine Initialization ---
        self._initialize_and_start_runtime(payload)

    def _initialize_and_start_runtime(self, start_payload: Dict[str, Any] = None):
        """Creates and starts a new SdeRuntimeEngine instance."""
        if start_payload is None:
            start_payload = {}
        try:
            challenge_name = self.experiment.challenge['name']
            challenge_def = AVAILABLE_DATASETS[challenge_name]

            scheduler_name = self.experiment.adaptive_policy
            scheduler_class = SCHEDULER_MAP.get(scheduler_name)
            if not scheduler_class:
                raise ValueError(f"Unknown scheduler '{scheduler_name}' specified in adaptive_policy.")

            increasing = "accuracy" in challenge_def.performance_metric_name.lower()

            # --- Instantiate the scheduler with correct parameters ---
            scheduler_args = {
                "metric": challenge_def.performance_metric_name,
                "increasing": increasing,
            }
            if scheduler_name == "Hyperband":
                max_resource = self.experiment.patience_budget.get('max_epochs', 81) if self.experiment.patience_budget else 81
                scheduler_args['max_resource_per_trial'] = max_resource

            scheduler = scheduler_class(**scheduler_args)
            # --- End of instantiation ---

            # Important: Pass a copy of the list of trials to the engine
            current_trials = list(self.experiment.trials.values())

            # Read execution settings from the payload
            enable_checkpointing = start_payload.get('enable_checkpointing', False)

            self.runtime_engine = SdeRuntimeEngine(
                trials=current_trials,
                dataset_name=challenge_name,
                adaptive_scheduler=scheduler,
                trial_updated_callback=self.on_trial_updated,
                insights_callback=self.on_insights_generated,
                enable_checkpointing=enable_checkpointing,
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
            self.runtime_engine.pause()
            self.experiment.status = ExperimentStatus.PAUSED
            self.log_message.emit("INFO: Experiment paused.")

    def handle_resume_run(self, payload: Dict[str, Any]):
        if self.runtime_engine:
            self.runtime_engine.resume()
            self.experiment.status = ExperimentStatus.RUNNING
            self.log_message.emit("INFO: Experiment resumed.")
        else:
            self.log_message.emit("ERROR: Cannot resume, no runtime engine exists. Please start the run first.")

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
            actions["global"].append("ADD_ALGORITHM")
            actions["global"].append("SET_ADAPTIVE_POLICY") # Allow changing policy mid-run
            actions["global"].append("SET_BUDGET") # Allow changing budget mid-run


        elif status == ExperimentStatus.PAUSED:
            actions["global"].append("RESUME_RUN")
            actions["global"].append("ADD_ALGORITHM")
            actions["global"].append("SET_ADAPTIVE_POLICY")
            actions["global"].append("SET_BUDGET")

        # Per-algorithm actions
        for algo_id, algo in self.experiment.algorithms.items():
            algo_actions = []
            if status == ExperimentStatus.DEFINING:
                algo_actions.append("UPDATE_PARAM_SPACE")
                algo_actions.append("REMOVE_ALGORITHM")
            elif status == ExperimentStatus.RUNNING or status == ExperimentStatus.PAUSED:
                algo_actions.append("UPDATE_PARAM_SPACE") # Allow updating space mid-run
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
