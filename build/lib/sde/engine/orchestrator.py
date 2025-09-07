import copy
import logging
import threading
import traceback
from typing import Any
from typing import Dict
from typing import List

from sde.core.actions import ActionType
from sde.core.types import AlgorithmConfig
from sde.core.types import Experiment
from sde.core.types import ExperimentStatus
from sde.core.types import Trial
from sde.core.types import TrialStatus
from sde.engine.action_validator import ActionValidator
from sde.engine.factory import SchedulerFactory
from sde.engine.runtime import SdeRuntimeEngine
from sde.events import Signal
from sde.persistence import load_experiment
from sde.persistence import save_experiment

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """The central nervous system of the SDE.

    This class manages the canonical `Experiment` state object, validates all
    incoming `Actions` from the UI, mutates the state, and issues high-level
    commands to the SdeRuntimeEngine. It is the sole source of truth for the
    application's state.
    """

    # Signals to update the UI
    state_changed = Signal(dict)
    log_message = Signal(dict)

    def __init__(self):
        self.experiment = Experiment()
        self.runtime_engine = None  # Will be initialized on START_RUN
        self._lock = threading.Lock()
        self.log_message.emit({'level': 'INFO', 'message': "Orchestrator initialized in DEFINING state."})

    def dispatch(self, action_type: ActionType, payload: Dict[str, Any]) -> None:
        """Receives an action, validates it, mutates the state, and triggers side effects.
        """
        with self._lock:
            handler_name = f"handle_{action_type.value.lower()}"
            handler = getattr(self, handler_name, None)
            if not handler:
                self.log_message.emit({'level': 'ERROR', 'message': f"No handler for action '{action_type.value}'"})
                return

            valid_actions = ActionValidator.get_valid_actions(self.experiment)
            if not ActionValidator.is_action_valid(action_type.value, payload, valid_actions):
                self.log_message.emit({
                    'level': 'WARN',
                    'message': f"Action '{action_type.value}' is not valid for the current state or payload."
                })
                return

            try:
                handler(payload)
                self.emit_state_change()
            except Exception as e:
                self.log_message.emit({
                    'level': 'ERROR',
                    'message': f"Failed to execute action {action_type.value}: {e}"
                })
                logger.error(traceback.format_exc())

    def emit_state_change(self) -> None:
        """Serializes the current experiment state and emits it via the state_changed signal.
        """
        state_dict = self.experiment.to_dict()
        state_dict["valid_actions"] = ActionValidator.get_valid_actions(self.experiment)
        self.state_changed.emit(state_dict)

    # --- Action Handlers ---

    def handle_set_challenge(self, payload: Dict[str, Any]) -> None:
        """Sets the challenge for the experiment."""
        self.experiment.challenge = payload
        self.log_message.emit({
            'level': 'INFO',
            'message': f"Challenge set to '{payload.get('name', 'Unknown')}'"
        })

    def handle_add_algorithm(self, payload: Dict[str, Any]) -> None:
        """Adds a new algorithm to the experiment, generating trials if the run is active."""
        algo_id = f"algo_{len(self.experiment.algorithms)}"
        new_algo = AlgorithmConfig(
            id=algo_id, name=payload["name"], parameter_space=payload["parameter_space"]
        )
        self.experiment.algorithms[algo_id] = new_algo
        self.log_message.emit({'level': 'INFO', 'message': f"Added algorithm: {new_algo.name}"})

        if self.experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            num_trials = payload.get("num_trials")
            if num_trials is None:
                self.log_message.emit({
                    'level': 'WARN',
                    'message': "Add algorithm mid-run requires 'num_trials' in payload. Skipping trial generation."
                })
                return

            self.log_message.emit({
                'level': 'INFO',
                'message': f"Adding {num_trials} new trials for algorithm '{new_algo.name}' mid-run."
            })
            self._generate_and_add_trials([new_algo], num_trials_per_algo=num_trials)

    def handle_remove_algorithm(self, payload: Dict[str, Any]) -> None:
        """Removes an algorithm and prunes all of its associated trials."""
        algo_id = payload["algorithm_id"]
        if algo_id in self.experiment.algorithms:
            algo_name = self.experiment.algorithms[algo_id].name
            del self.experiment.algorithms[algo_id]
            for trial in self.experiment.trials.values():
                if trial.algorithm_name == algo_name:
                    trial.status = TrialStatus.PRUNED
                    if self.runtime_engine:
                        self.runtime_engine.cancel_work_for_trial(trial.id)
            self.log_message.emit({
                'level': 'INFO',
                'message': f"Removed algorithm '{algo_name}' and pruned its trials."
            })
        else:
            self.log_message.emit({
                'level': 'WARN',
                'message': f"Could not find algorithm with id {algo_id} to remove."
            })

    def handle_update_param_space(self, payload: Dict[str, Any]) -> None:
        """Updates the hyperparameter space for an algorithm, generating new trials if active."""
        algo_id = payload["algorithm_id"]
        new_space = payload["new_space"]
        if algo_id not in self.experiment.algorithms:
            self.log_message.emit({
                'level': 'WARN',
                'message': f"Could not find algorithm with id {algo_id} to update."
            })
            return
        algo = self.experiment.algorithms[algo_id]
        algo.parameter_space = new_space
        self.log_message.emit({
            'level': 'INFO',
            'message': f"Updated parameter space for algorithm {algo.name} ({algo_id})."
        })
        if self.runtime_engine and self.experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.log_message.emit({
                'level': 'INFO',
                'message': f"Generating new trials for '{algo.name}' due to parameter space update."
            })
            # Re-use the original number of trials per algorithm from the execution settings
            num_trials = self.experiment.execution_settings.get("num_trials_per_algo", 10)
            self._generate_and_add_trials([algo], num_trials_per_algo=num_trials)

    def handle_set_adaptive_policy(self, payload: Dict[str, Any]) -> None:
        """Sets the adaptive scheduling policy for the experiment."""
        policy_name = payload["policy_name"]
        self.experiment.adaptive_policy = policy_name
        self.log_message.emit({'level': 'INFO', 'message': f"Adaptive policy set to '{policy_name}'."})
        if self.runtime_engine and self.experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.log_message.emit({'level': 'INFO', 'message': "Hot-swapping adaptive policy in live runtime engine."})
            try:
                new_scheduler = SchedulerFactory.create_scheduler(
                    policy_name=policy_name,
                    challenge_name=self.experiment.challenge["name"],
                    patience_budget=self.experiment.patience_budget,
                )
                self.runtime_engine.update_adaptive_policy(new_scheduler)
                self.log_message.emit({'level': 'INFO', 'message': "Adaptive policy updated successfully in runtime."})
            except Exception as e:
                self.log_message.emit({'level': 'ERROR', 'message': f"Failed to hot-swap adaptive policy: {e}"})
                logger.error(f"Failed to hot-swap adaptive policy: {traceback.format_exc()}")

    def handle_set_budget(self, payload: Dict[str, Any]) -> None:
        """Sets the patience budget for the experiment."""
        self.experiment.patience_budget = payload
        self.log_message.emit({'level': 'INFO', 'message': f"Patience budget set to {payload}."})

    def handle_manual_prune_trial(self, payload: Dict[str, Any]) -> None:
        """Manually prunes a single trial, stopping any active work."""
        trial_id = payload["trial_id"]
        if trial_id in self.experiment.trials:
            self.experiment.trials[trial_id].status = TrialStatus.PRUNED
            if self.runtime_engine:
                self.runtime_engine.cancel_work_for_trial(trial_id)
            self.log_message.emit({'level': 'INFO', 'message': f"Manually pruned trial {trial_id}."})
        else:
            self.log_message.emit({'level': 'WARN', 'message': f"Could not find trial with id {trial_id} to prune."})

    def handle_manual_prioritize_trial(self, payload: Dict[str, Any]) -> None:
        """Manually increases the priority of a single trial."""
        trial_id = payload["trial_id"]
        if trial_id in self.experiment.trials:
            self.experiment.trials[trial_id].priority += 10
            self.log_message.emit({'level': 'INFO', 'message': f"Increased priority for trial {trial_id}."})
        else:
            self.log_message.emit({'level': 'WARN', 'message': f"Could not find trial with id {trial_id} to prioritize."})

    def handle_spawn_similar_trial(self, payload: Dict[str, Any]) -> None:
        """Creates a new trial based on an existing one, with optional new hyperparameters."""
        source_trial_id = payload["source_trial_id"]
        if source_trial_id not in self.experiment.trials:
            self.log_message.emit({'level': 'WARN', 'message': f"Could not find source trial {source_trial_id} to spawn from."})
            return
        source_trial = self.experiment.trials[source_trial_id]
        new_hparams = payload.get("new_hparams", source_trial.hyperparameters.copy())
        new_trial_id = f"trial_{source_trial.algorithm_name.lower()}_{len(self.experiment.trials)}"
        new_trial = Trial(
            id=new_trial_id,
            algorithm_name=source_trial.algorithm_name,
            hyperparameters=new_hparams,
            status=TrialStatus.PENDING,
        )
        self.experiment.trials[new_trial_id] = new_trial
        if self.runtime_engine:
            self.runtime_engine.add_trials_live([new_trial])
        self.log_message.emit({'level': 'INFO', 'message': f"Spawned new trial {new_trial_id} from {source_trial_id}."})

    def handle_start_run(self, payload: Dict[str, Any]) -> None:
        """Starts the experiment run by initializing and starting the runtime engine."""
        self.log_message.emit({'level': 'INFO', 'message': "START_RUN action received. Validating and initializing runtime."})
        if not self.experiment.algorithms:
            self.log_message.emit({'level': 'ERROR', 'message': "Cannot start run without at least one algorithm."})
            return
        self.experiment.status = ExperimentStatus.RUNNING
        self.experiment.execution_settings = payload
        if not self.experiment.trials:
            self.log_message.emit({'level': 'INFO', 'message': "No pre-existing trials found. Generating initial set."})
            num_trials = payload["num_trials_per_algo"]
            algorithms = list(self.experiment.algorithms.values())
            success = self._generate_and_add_trials(algorithms, num_trials)
            if not success:
                self.experiment.status = ExperimentStatus.DEFINING
                return
        self._initialize_and_start_runtime()

    def _generate_and_add_trials(self, algorithms: List[Any], num_trials_per_algo: int) -> bool:
        """Generates a set of trials for the given algorithms using the adaptive scheduler."""
        self.log_message.emit({
            'level': 'INFO',
            'message': f"Generating {num_trials_per_algo} trials for {len(algorithms)} algorithm(s)."
        })
        try:
            if self.runtime_engine and self.runtime_engine.adaptive_scheduler:
                scheduler = self.runtime_engine.adaptive_scheduler
            else:
                scheduler = SchedulerFactory.create_scheduler(
                    policy_name=self.experiment.adaptive_policy,
                    challenge_name=self.experiment.challenge["name"],
                    patience_budget=self.experiment.patience_budget,
                )
            if not scheduler:
                self.log_message.emit({'level': 'ERROR', 'message': "Could not create or find a scheduler."})
                return False
            new_trials = scheduler.generate_initial_trials(algorithms, num_trials_per_algo)
            for trial in new_trials:
                self.experiment.trials[trial.id] = trial
            if self.runtime_engine:
                self.runtime_engine.add_trials_live(new_trials)
            self.log_message.emit({'level': 'INFO', 'message': f"Added {len(new_trials)} new trials to the experiment."})
            return True
        except Exception as e:
            self.log_message.emit({'level': 'ERROR', 'message': f"Failed to generate or add new trials: {e}"})
            logger.error(f"Trial generation/addition failed: {traceback.format_exc()}")
            return False

    def _initialize_and_start_runtime(self, start_paused: bool = False) -> None:
        """Creates, configures, and starts the SdeRuntimeEngine in a background thread."""
        execution_settings = self.experiment.execution_settings or {}
        try:
            challenge_name = self.experiment.challenge["name"]
            scheduler = SchedulerFactory.create_scheduler(
                policy_name=self.experiment.adaptive_policy,
                challenge_name=challenge_name,
                patience_budget=self.experiment.patience_budget,
            )
            current_trials = list(self.experiment.trials.values())
            enable_checkpointing = execution_settings.get("enable_checkpointing", False)
            num_workers = execution_settings.get("num_workers", 1)

            self.runtime_engine = SdeRuntimeEngine(
                trials=current_trials,
                dataset_name=challenge_name,
                adaptive_scheduler=scheduler,
                trial_updated_callback=self.on_trial_updated,
                insights_callback=self.on_insights_generated,
                enable_checkpointing=enable_checkpointing,
                max_workers=num_workers,
            )
            self.log_message.emit({
                'level': 'INFO',
                'message': f"SdeRuntimeEngine initialized with {self.experiment.adaptive_policy} scheduler."
            })
            self.runtime_engine.start(start_paused=start_paused)
            self.log_message.emit({'level': 'INFO', 'message': "SdeRuntimeEngine started successfully."})
        except Exception as e:
            self.log_message.emit({'level': 'ERROR', 'message': f"Failed to start runtime engine: {e}"})
            logger.error(f"Engine start failed: {traceback.format_exc()}")
            self.experiment.status = ExperimentStatus.DEFINING

    def on_trial_updated(self, trial_data: Dict[str, Any]) -> None:
        """Callback executed by the runtime when a trial's state has changed."""
        with self._lock:
            trial_id = trial_data.get("id")
            if not trial_id:
                logger.warning("Orchestrator received update without a trial_id.")
                return
            updated_trial = Trial.from_dict(trial_data)
            self.experiment.trials[updated_trial.id] = updated_trial
        self.emit_state_change()

    def on_insights_generated(self, insights: List[Dict[str, Any]]) -> None:
        """Callback executed by the runtime when new insights are available."""
        with self._lock:
            self.experiment.insights.extend(insights)
            for insight in insights:
                self.log_message.emit({'level': 'INSIGHT', 'message': insight['message'], 'data': insight})
        self.emit_state_change()

    def handle_pause_run(self, payload: Dict[str, Any]) -> None:
        """Pauses the current experiment run."""
        if self.runtime_engine:
            self.runtime_engine.pause()
            self.experiment.status = ExperimentStatus.PAUSED
            self.log_message.emit({'level': 'INFO', 'message': "Experiment paused."})

    def handle_resume_run(self, payload: Dict[str, Any]) -> None:
        """Resumes a paused experiment run."""
        if self.runtime_engine:
            self.runtime_engine.resume()
            self.experiment.status = ExperimentStatus.RUNNING
            self.log_message.emit({'level': 'INFO', 'message': "Experiment resumed."})
        else:
            self.log_message.emit({'level': 'ERROR', 'message': "Cannot resume, no runtime engine exists. Please start the run first."})

    def handle_save_experiment(self, payload: Dict[str, Any]) -> None:
        """Handles the request to save the current experiment state to a file."""
        filepath = payload.get("filepath")
        if not filepath:
            self.log_message.emit({'level': 'ERROR', 'message': "No filepath provided for saving experiment."})
            return
        try:
            experiment_copy = copy.deepcopy(self.experiment)
            save_experiment(experiment_copy, filepath)
            self.log_message.emit({'level': 'INFO', 'message': f"Experiment successfully saved to {filepath}"})
        except Exception as e:
            self.log_message.emit({'level': 'ERROR', 'message': f"Failed to save experiment: {e}"})
            logger.error(f"Failed to save experiment: {traceback.format_exc()}")

    def handle_load_experiment(self, payload: Dict[str, Any]) -> None:
        """Handles the request to load an experiment state from a file."""
        filepath = payload.get("filepath")
        if not filepath:
            self.log_message.emit({'level': 'ERROR', 'message': "No filepath provided for loading experiment."})
            return
        if self.runtime_engine:
            self.shutdown()
            self.runtime_engine = None
        try:
            self.experiment = load_experiment(filepath)
            self.log_message.emit({'level': 'INFO', 'message': f"Experiment successfully loaded from {filepath}."})

            # If the loaded experiment was paused, re-initialize the engine in a paused state
            if self.experiment.status == ExperimentStatus.PAUSED:
                self.log_message.emit({'level': 'INFO', 'message': "Restoring paused experiment. Initializing runtime engine..."})
                # This will create and start the engine, but the engine's loop will be
                # immediately blocked because we pass start_paused=True.
                self._initialize_and_start_runtime(start_paused=True)
                # The status is set to RUNNING inside start, so we set it back to PAUSED
                self.experiment.status = ExperimentStatus.PAUSED
                self.log_message.emit({'level': 'INFO', 'message': "Engine is ready. Press 'Resume' to continue the run."})
            else:
                # For COMPLETED or other states, just load and let the user inspect.
                self.log_message.emit({'level': 'INFO', 'message': f"Loaded experiment is in '{self.experiment.status.value}' state."})

        except Exception as e:
            self.log_message.emit({'level': 'ERROR', 'message': f"Failed to load experiment: {e}"})
            logger.error(f"Failed to load experiment: {traceback.format_exc()}")
            self.experiment = Experiment()

    def shutdown(self) -> None:
        """Gracefully shuts down the runtime engine if it exists."""
        if self.runtime_engine:
            self.log_message.emit({'level': 'INFO', 'message': "Orchestrator shutting down runtime engine..."})
            self.runtime_engine.stop()
            self.log_message.emit({'level': 'INFO', 'message': "Runtime engine shut down."})
