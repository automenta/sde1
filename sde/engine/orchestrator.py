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

            # Use the validator to check if the action is allowed
            if action_type not in self.get_valid_actions():
                self.log_message.emit(f"WARN: Action '{action_type}' is not valid in state '{self.experiment.status.value}'")
                return

            try:
                handler(payload)
                self.emit_state_change()
            except Exception as e:
                self.log_message.emit(f"ERROR: Failed to execute action {action_type}: {e}")
                logger.error(traceback.format_exc())

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

    def handle_start_run(self, payload: Dict[str, Any]):
        self.log_message.emit("INFO: START_RUN action received. Validating and initializing runtime.")
        if not self.experiment.algorithms:
            self.log_message.emit("ERROR: Cannot start run without at least one algorithm.")
            return

        self.experiment.status = ExperimentStatus.RUNNING

        # --- Trial Generation (for simple runs) ---
        if not self.experiment.trials:
            self.log_message.emit("INFO: No pre-existing trials found. Generating default trials.")
            for algo_config in self.experiment.algorithms.values():
                # For a simple run, sample one set of hyperparameters.
                # We'll just take the midpoint of the defined parameter space.
                hparams = {}
                for p_name, p_space in algo_config.parameter_space.items():
                    if isinstance(p_space, tuple) and len(p_space) == 2:
                        # Assuming (min, max) for numeric types
                        hparams[p_name] = (p_space[0] + p_space[1]) / 2
                    else:
                        # Fallback for non-numeric or fixed value spaces
                        hparams[p_name] = p_space

                trial_id = f"trial_{algo_config.name.lower().replace(' ', '_')}_{len(self.experiment.trials)}"
                trial = Trial(
                    id=trial_id,
                    algorithm_name=algo_config.name,
                    hyperparameters=hparams,
                )
                self.experiment.trials[trial_id] = trial
            self.log_message.emit(f"INFO: Generated {len(self.experiment.trials)} initial trials.")

        # --- Engine Initialization ---
        try:
            challenge_name = self.experiment.challenge['name']
            challenge_def = AVAILABLE_DATASETS[challenge_name]

            scheduler_name = self.experiment.adaptive_policy
            scheduler_class = SCHEDULER_MAP.get(scheduler_name)
            if not scheduler_class:
                raise ValueError(f"Unknown scheduler '{scheduler_name}' specified in adaptive_policy.")

            # Note: A more robust solution would define this in the challenge definition itself.
            # For now, we infer it based on common metric names.
            increasing = "accuracy" in challenge_def.performance_metric_name.lower()

            # Instantiate the scheduler
            # This assumes schedulers have a compatible signature. A factory pattern
            # would be more robust for schedulers with different needs.
            scheduler = scheduler_class(metric=challenge_def.performance_metric_name, increasing=increasing)

            self.runtime_engine = SdeRuntimeEngine(
                trials=list(self.experiment.trials.values()),
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
        if self.runtime_engine:
            # The existing engine was designed to be started once.
            # A more robust implementation would re-create it or ensure
            # its internal state is ready for a restart. For now, we assume
            # the _execution_loop can be re-entered.
            self.runtime_engine.start()
            self.experiment.status = ExperimentStatus.RUNNING
            self.log_message.emit("INFO: Experiment resumed.")

    def get_valid_actions(self) -> List[str]:
        """
        This is the Action Validator. It inspects the current state and returns
        a list of action types that are currently valid.
        """
        status = self.experiment.status
        actions = []

        if status == ExperimentStatus.DEFINING:
            if self.experiment.challenge is None:
                actions.append("SET_CHALLENGE")
            else:
                actions.append("ADD_ALGORITHM")
                if self.experiment.algorithms:
                    actions.append("START_RUN")

        elif status == ExperimentStatus.RUNNING:
            actions.append("PAUSE_RUN")
            actions.append("ADD_ALGORITHM") # Example of a mid-run interaction

        elif status == ExperimentStatus.PAUSED:
            actions.append("RESUME_RUN")

        return actions
