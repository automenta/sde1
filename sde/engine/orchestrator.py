import logging
import traceback
import threading
import copy
from typing import List, Dict, Any

from sde.core.types import Experiment, ExperimentStatus, Trial, TrialStatus
from sde.core.actions import ActionType
from sde.engine.runtime import SdeRuntimeEngine
from sde.challenges import AVAILABLE_DATASETS
from sde.engine.action_validator import ActionValidator
from sde.engine.factory import SchedulerFactory
from sde.events import Signal
from sde.persistence import save_experiment, load_experiment
from sde import config


logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """
    The central nervous system of the SDE.

    This class manages the canonical `Experiment` state object, validates all
    incoming `Actions` from the UI, mutates the state, and issues high-level
    commands to the SdeRuntimeEngine. It is the sole source of truth for the
    application's state.
    """

    # Signals to update the UI
    state_changed = Signal(dict)
    log_message = Signal(str)

    def __init__(self):
        self.experiment = Experiment()
        self.runtime_engine = None  # Will be initialized on START_RUN
        self._lock = threading.Lock()
        self.log_message.emit("INFO: Orchestrator initialized in DEFINING state.")

    def dispatch(self, action_type: ActionType, payload: Dict[str, Any]) -> None:
        """
        Receives an action, validates it, mutates the state, and triggers side effects.

        This is the main entry point for all UI-driven changes to the experiment state.

        Args:
            action_type: The type of the action to be processed.
            payload: The data associated with the action.
        """
        with self._lock:
            handler_name = f"handle_{action_type.value.lower()}"
            handler = getattr(self, handler_name, None)
            if not handler:
                self.log_message.emit(f"ERROR: No handler for action '{action_type.value}'")
                return

            # Use the new structured validator
            valid_actions = ActionValidator.get_valid_actions(self.experiment)
            if not ActionValidator.is_action_valid(
                action_type.value, payload, valid_actions
            ):
                self.log_message.emit(
                    f"WARN: Action '{action_type.value}' is not valid for the current state or payload."
                )
                return

            try:
                handler(payload)
                self.emit_state_change()
            except Exception as e:
                self.log_message.emit(
                    f"ERROR: Failed to execute action {action_type.value}: {e}"
                )
                logger.error(traceback.format_exc())

    def emit_state_change(self) -> None:
        """
        Serializes the current experiment state and emits it via the state_changed signal.

        This method is the canonical way to notify the UI of any changes.
        It also enriches the state with dynamic information, like the set of
        currently valid actions.
        """
        # Use the new to_dict() method for robust serialization
        state_dict = self.experiment.to_dict()
        # Add dynamic information that is not part of the core experiment state
        state_dict["valid_actions"] = ActionValidator.get_valid_actions(
            self.experiment
        )
        self.state_changed.emit(state_dict)

    # --- Action Handlers ---

    def handle_set_challenge(self, payload: Dict[str, Any]) -> None:
        """
        Sets the scientific challenge for the experiment.

        Args:
            payload: A dictionary representing the challenge configuration,
                     e.g., {'name': 'CIFAR10'}.
        """
        self.experiment.challenge = payload
        self.log_message.emit(
            f"INFO: Challenge set to '{payload.get('name', 'Unknown')}'"
        )

    def handle_add_algorithm(self, payload: Dict[str, Any]) -> None:
        """
        Adds a new algorithm to the experiment.

        If the experiment is already running, this will trigger the generation
        of new trials for this algorithm, which are added to the live run.

        Args:
            payload: A dictionary containing the algorithm 'name' and its 'parameter_space'.
        """
        from sde.core.types import AlgorithmConfig

        algo_id = f"algo_{len(self.experiment.algorithms)}"
        new_algo = AlgorithmConfig(
            id=algo_id, name=payload["name"], parameter_space=payload["parameter_space"]
        )
        self.experiment.algorithms[algo_id] = new_algo
        self.log_message.emit(f"INFO: Added algorithm: {new_algo.name}")

        # If the experiment is already running, generate trials and add them live
        if self.experiment.status in [
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
        ]:
            self.log_message.emit(
                f"INFO: Adding new trials for algorithm '{new_algo.name}' mid-run."
            )
            self._generate_and_add_trials([new_algo])

    def handle_remove_algorithm(self, payload: Dict[str, Any]) -> None:
        """
        Removes an algorithm from the experiment.

        This also finds all associated trials and marks them as 'PRUNED'.
        If the experiment is running, it cancels any pending work for those trials.

        Args:
            payload: A dictionary containing the 'algorithm_id' to remove.
        """
        algo_id = payload["algorithm_id"]
        if algo_id in self.experiment.algorithms:
            algo_name = self.experiment.algorithms[algo_id].name
            del self.experiment.algorithms[algo_id]

            # Prune associated trials
            for trial in self.experiment.trials.values():
                if trial.algorithm_name == algo_name:
                    trial.status = TrialStatus.PRUNED
                    if self.runtime_engine:
                        self.runtime_engine.cancel_work_for_trial(trial.id)

            self.log_message.emit(
                f"INFO: Removed algorithm '{algo_name}' and pruned its trials."
            )
        else:
            self.log_message.emit(
                f"WARN: Could not find algorithm with id {algo_id} to remove."
            )

    def handle_update_param_space(self, payload: Dict[str, Any]) -> None:
        """
        Updates the hyperparameter space for a given algorithm.

        If the experiment is running, this triggers the generation of new trials
        based on the updated parameter space.

        Args:
            payload: A dictionary containing the 'algorithm_id' and the 'new_space'.
        """
        algo_id = payload["algorithm_id"]
        new_space = payload["new_space"]
        if algo_id not in self.experiment.algorithms:
            self.log_message.emit(
                f"WARN: Could not find algorithm with id {algo_id} to update."
            )
            return

        algo = self.experiment.algorithms[algo_id]
        algo.parameter_space = new_space
        self.log_message.emit(
            f"INFO: Updated parameter space for algorithm {algo.name} ({algo_id})."
        )

        # If the experiment is running, generate new trials based on the updated space
        if self.runtime_engine and self.experiment.status in [
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
        ]:
            self.log_message.emit(
                f"INFO: Generating new trials for '{algo.name}' due to parameter space update."
            )
            self._generate_and_add_trials([algo])

    def handle_set_adaptive_policy(self, payload: Dict[str, Any]) -> None:
        """
        Sets the adaptive scheduling policy for the experiment.

        If the experiment is running, this will "hot-swap" the scheduler
        in the live runtime engine.

        Args:
            payload: A dictionary containing the 'policy_name'.
        """
        policy_name = payload["policy_name"]
        # The factory will raise an error if the policy name is invalid,
        # which is caught by the main dispatch loop.
        self.experiment.adaptive_policy = policy_name
        self.log_message.emit(f"INFO: Adaptive policy set to '{policy_name}'.")

        # If the run is live, hot-swap the scheduler in the runtime engine
        if self.runtime_engine and self.experiment.status in [
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
        ]:
            self.log_message.emit(
                "INFO: Hot-swapping adaptive policy in live runtime engine."
            )
            try:
                new_scheduler = SchedulerFactory.create_scheduler(
                    policy_name=policy_name,
                    challenge_name=self.experiment.challenge["name"],
                    patience_budget=self.experiment.patience_budget,
                )
                self.runtime_engine.update_adaptive_policy(new_scheduler)
                self.log_message.emit(
                    "INFO: Adaptive policy updated successfully in runtime."
                )
            except Exception as e:
                self.log_message.emit(f"ERROR: Failed to hot-swap adaptive policy: {e}")
                logger.error(
                    f"Failed to hot-swap adaptive policy: {traceback.format_exc()}"
                )

    def handle_set_budget(self, payload: Dict[str, Any]) -> None:
        """
        Sets the patience budget for the experiment.

        Args:
            payload: A dictionary defining the budget, e.g., {'max_epochs': 100}.
        """
        self.experiment.patience_budget = payload
        self.log_message.emit(f"INFO: Patience budget set to {payload}.")

    def handle_manual_prune_trial(self, payload: Dict[str, Any]) -> None:
        """
        Manually prunes a specific trial.

        Args:
            payload: A dictionary containing the 'trial_id' to prune.
        """
        trial_id = payload["trial_id"]
        if trial_id in self.experiment.trials:
            self.experiment.trials[trial_id].status = TrialStatus.PRUNED
            if self.runtime_engine:
                self.runtime_engine.cancel_work_for_trial(trial_id)
            self.log_message.emit(f"INFO: Manually pruned trial {trial_id}.")
        else:
            self.log_message.emit(
                f"WARN: Could not find trial with id {trial_id} to prune."
            )

    def handle_manual_prioritize_trial(self, payload: Dict[str, Any]) -> None:
        """
        Manually increases the priority of a specific trial.

        Args:
            payload: A dictionary containing the 'trial_id' to prioritize.
        """
        trial_id = payload["trial_id"]
        if trial_id in self.experiment.trials:
            # Increase priority by a fixed amount
            self.experiment.trials[trial_id].priority += 10
            self.log_message.emit(f"INFO: Increased priority for trial {trial_id}.")
        else:
            self.log_message.emit(
                f"WARN: Could not find trial with id {trial_id} to prioritize."
            )

    def handle_spawn_similar_trial(self, payload: Dict[str, Any]) -> None:
        """
        Spawns a new trial based on an existing one.

        This is useful for exploring variations of a promising trial. The new
        trial can have the same or slightly modified hyperparameters.

        Args:
            payload: A dictionary containing the 'source_trial_id' and an
                     optional 'new_hparams' dictionary.
        """
        source_trial_id = payload["source_trial_id"]
        if source_trial_id not in self.experiment.trials:
            self.log_message.emit(
                f"WARN: Could not find source trial {source_trial_id} to spawn from."
            )
            return

        source_trial = self.experiment.trials[source_trial_id]
        new_hparams = payload.get("new_hparams", source_trial.hyperparameters.copy())

        new_trial_id = (
            f"trial_{source_trial.algorithm_name.lower()}_{len(self.experiment.trials)}"
        )
        new_trial = Trial(
            id=new_trial_id,
            algorithm_name=source_trial.algorithm_name,
            hyperparameters=new_hparams,
            status=TrialStatus.PENDING,
        )
        self.experiment.trials[new_trial_id] = new_trial

        if self.runtime_engine:
            # Use the new, more general method for adding trials live
            self.runtime_engine.add_trials_live([new_trial])

        self.log_message.emit(
            f"INFO: Spawned new trial {new_trial_id} from {source_trial_id}."
        )

    def handle_start_run(self, payload: Dict[str, Any]) -> None:
        """
        Starts the experiment run.

        This initializes the SdeRuntimeEngine in a background thread and begins
        executing work units. If no trials exist, it generates the initial set.

        Args:
            payload: A dictionary that can contain execution settings like
                     'enable_checkpointing'.
        """
        self.log_message.emit(
            "INFO: START_RUN action received. Validating and initializing runtime."
        )
        if not self.experiment.algorithms:
            self.log_message.emit(
                "ERROR: Cannot start run without at least one algorithm."
            )
            return

        self.experiment.status = ExperimentStatus.RUNNING

        # --- V2 Trial Generation (Delegated) ---
        if not self.experiment.trials:
            self.log_message.emit(
                "INFO: No pre-existing trials found. Generating initial set."
            )
            algorithms = list(self.experiment.algorithms.values())
            success = self._generate_and_add_trials(algorithms)
            if not success:
                self.experiment.status = ExperimentStatus.DEFINING  # Revert status
                return

        # --- Engine Initialization ---
        self._initialize_and_start_runtime(payload)

    def _generate_and_add_trials(self, algorithms: List[Any], num_trials_per_algo: int = config.NUM_TRIALS_PER_ALGO) -> bool:
        """
        Generates new trials and adds them to the experiment.

        Delegates trial generation to the adaptive scheduler. If the runtime
        is live, the new trials are added to it immediately.

        Args:
            algorithms: A list of AlgorithmConfig objects to generate trials for.
            num_trials_per_algo: The number of trials to generate per algorithm.

        Returns:
            True if trials were generated and added successfully, False otherwise.
        """
        self.log_message.emit(
            f"INFO: Generating {num_trials_per_algo} trials for {len(algorithms)} algorithm(s)."
        )
        try:
            # Get the adaptive scheduler, creating it if it doesn't exist.
            # This is cleaner than the previous nested conditional logic.
            if self.runtime_engine and self.runtime_engine.adaptive_scheduler:
                scheduler = self.runtime_engine.adaptive_scheduler
            else:
                scheduler = SchedulerFactory.create_scheduler(
                    policy_name=self.experiment.adaptive_policy,
                    challenge_name=self.experiment.challenge["name"],
                    patience_budget=self.experiment.patience_budget,
                )

            if not scheduler:
                 self.log_message.emit("ERROR: Could not create or find a scheduler.")
                 return False

            new_trials = scheduler.generate_initial_trials(
                algorithms, num_trials_per_algo
            )

            for trial in new_trials:
                self.experiment.trials[trial.id] = trial

            if self.runtime_engine:
                self.runtime_engine.add_trials_live(new_trials)

            self.log_message.emit(
                f"INFO: Added {len(new_trials)} new trials to the experiment."
            )
            return True

        except Exception as e:
            self.log_message.emit(f"ERROR: Failed to generate or add new trials: {e}")
            logger.error(
                f"Trial generation/addition failed: {traceback.format_exc()}"
            )
            return False

    def _initialize_and_start_runtime(self, start_payload: Dict[str, Any] = None) -> None:
        """
        Creates, configures, and starts a new SdeRuntimeEngine instance.

        This is the primary method for kicking off the background computational
        processes.

        Args:
            start_payload: Optional dictionary with runtime settings, e.g.,
                           {'enable_checkpointing': True}.
        """
        if start_payload is None:
            start_payload = {}
        try:
            challenge_name = self.experiment.challenge["name"]

            scheduler = SchedulerFactory.create_scheduler(
                policy_name=self.experiment.adaptive_policy,
                challenge_name=challenge_name,
                patience_budget=self.experiment.patience_budget,
            )

            # Important: Pass a copy of the list of trials to the engine
            current_trials = list(self.experiment.trials.values())

            # Read execution settings from the payload
            enable_checkpointing = start_payload.get("enable_checkpointing", False)

            self.runtime_engine = SdeRuntimeEngine(
                trials=current_trials,
                dataset_name=challenge_name,
                adaptive_scheduler=scheduler,
                trial_updated_callback=self.on_trial_updated,
                insights_callback=self.on_insights_generated,
                enable_checkpointing=enable_checkpointing,
            )

            self.log_message.emit(
                f"INFO: SdeRuntimeEngine initialized with {self.experiment.adaptive_policy} scheduler."
            )
            self.runtime_engine.start()
            self.log_message.emit("INFO: SdeRuntimeEngine started successfully.")

        except Exception as e:
            self.log_message.emit(f"ERROR: Failed to start runtime engine: {e}")
            logger.error(f"Engine start failed: {traceback.format_exc()}")
            self.experiment.status = (
                ExperimentStatus.DEFINING
            )  # Revert status on failure

    def on_trial_updated(self, trial_data: Dict[str, Any]) -> None:
        """
        Callback for the SdeRuntimeEngine to update the orchestrator's state.

        This method is called from the SDE Engine's background thread, so all
        access to shared state (self.experiment) must be protected by a lock.

        Args:
            trial_data: A dictionary representing the serialized state of a
                        single, updated trial.
        """
        with self._lock:
            trial_id = trial_data.get("id")
            if not trial_id:
                logger.warning(f"Orchestrator received update without a trial_id.")
                return

            # Use the new robust deserialization method
            updated_trial = Trial.from_dict(trial_data)
            self.experiment.trials[updated_trial.id] = updated_trial

        # Emit state change to notify UI
        self.emit_state_change()

    def on_insights_generated(self, insights: List[Dict[str, Any]]) -> None:
        """
        Callback for the SdeRuntimeEngine to add new insights to the state.

        This method is called from the SDE Engine's background thread.

        Args:
            insights: A list of new insights, each represented as a dictionary.
        """
        with self._lock:
            self.experiment.insights.extend(insights)
            for insight in insights:
                self.log_message.emit(f"INSIGHT: {insight['message']}")
        self.emit_state_change()

    def handle_pause_run(self, payload: Dict[str, Any]) -> None:
        """
        Pauses the currently running experiment.

        Args:
            payload: This argument is currently unused.
        """
        if self.runtime_engine:
            self.runtime_engine.pause()
            self.experiment.status = ExperimentStatus.PAUSED
            self.log_message.emit("INFO: Experiment paused.")

    def handle_resume_run(self, payload: Dict[str, Any]) -> None:
        """
        Resumes a paused experiment.

        Args:
            payload: This argument is currently unused.
        """
        if self.runtime_engine:
            self.runtime_engine.resume()
            self.experiment.status = ExperimentStatus.RUNNING
            self.log_message.emit("INFO: Experiment resumed.")
        else:
            self.log_message.emit(
                "ERROR: Cannot resume, no runtime engine exists. Please start the run first."
            )

    def handle_save_experiment(self, payload: Dict[str, Any]) -> None:
        """
        Handles the request to save the current experiment state to a file.

        Args:
            payload: A dictionary containing the 'filepath' for saving.
        """
        filepath = payload.get("filepath")
        if not filepath:
            self.log_message.emit("ERROR: No filepath provided for saving experiment.")
            return

        try:
            # It's safest to save a deep copy to avoid any potential race conditions
            # with the background thread, though the lock provides protection.
            experiment_copy = copy.deepcopy(self.experiment)
            save_experiment(experiment_copy, filepath)
            self.log_message.emit(f"INFO: Experiment successfully saved to {filepath}")
        except Exception as e:
            self.log_message.emit(f"ERROR: Failed to save experiment: {e}")
            logger.error(f"Failed to save experiment: {traceback.format_exc()}")

    def handle_load_experiment(self, payload: Dict[str, Any]) -> None:
        """
        Handles the request to load an experiment state from a file.

        This will stop any currently running experiment before loading the new one.

        Args:
            payload: A dictionary containing the 'filepath' for loading.
        """
        filepath = payload.get("filepath")
        if not filepath:
            self.log_message.emit("ERROR: No filepath provided for loading experiment.")
            return

        # First, stop any existing runtime engine
        if self.runtime_engine:
            self.shutdown()
            self.runtime_engine = None

        try:
            self.experiment = load_experiment(filepath)
            # The loaded experiment might be in a running or paused state.
            # For simplicity, we'll reset it to DEFINING to avoid complex state
            # reconstruction of the runtime. A more advanced implementation
            # might try to reconstruct and resume the SdeRuntimeEngine.
            self.experiment.status = ExperimentStatus.DEFINING
            self.log_message.emit(f"INFO: Experiment successfully loaded from {filepath}.")
            self.log_message.emit("INFO: Loaded experiment state has been reset to 'DEFINING'. Please press 'Start Run' to resume.")
        except Exception as e:
            self.log_message.emit(f"ERROR: Failed to load experiment: {e}")
            logger.error(f"Failed to load experiment: {traceback.format_exc()}")
            # Reset to a clean state on failure
            self.experiment = Experiment()

    def shutdown(self) -> None:
        """
        Safely shuts down the runtime engine.

        This should be called when the application is closing to ensure that
        background processes are terminated cleanly.
        """
        if self.runtime_engine:
            self.log_message.emit("INFO: Orchestrator shutting down runtime engine...")
            self.runtime_engine.stop()
            self.log_message.emit("INFO: Runtime engine shut down.")
