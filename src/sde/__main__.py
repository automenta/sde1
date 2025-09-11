import logging
import threading
import time
from typing import Dict

from .core.actions import ActionType
from .core.domain import ExperimentStatus
from .engine.orchestrator import ExperimentOrchestrator


def main():
    """Main entry point for running a command-line-based SDE experiment."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("sde_cli")
    start_time = time.time()

    # --- Experiment Configuration ---
    challenge_config = {
        "name": "MNIST",
        "description": "70,000 28x28 grayscale images of handwritten digits (0-9).",
    }
    algorithms_to_run = [
        {
            "name": "SimpleCNN",
            "parameter_space": {
                "dropout_rate": {"type": "float", "min": 0.1, "max": 0.5},
                "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
            }
        },
        {
            "name": "LogisticRegression",
            "parameter_space": {
                 "learning_rate": {"type": "float", "min": 1e-3, "max": 1e-1, "log": True},
            }
        },
    ]
    execution_settings = {
        "num_trials_per_algo": 4,
        "num_workers": 2,
        "enable_checkpointing": True,
        "work_unit_timeout_seconds": 300,
    }
    # --- End Configuration ---

    orchestrator = ExperimentOrchestrator()
    experiment_done = threading.Event()

    def on_state_changed(state: Dict):
        """Callback to handle state changes from the orchestrator."""
        status = state.get("status")
        if status in [
            ExperimentStatus.COMPLETED.value,
            ExperimentStatus.FAILED.value,
        ]:
            logger.info(f"Experiment finished with status: {status}. Shutting down.")
            experiment_done.set()

    def on_log_message(msg: Dict):
        """Callback to handle log messages."""
        level = msg.get("level", "INFO").upper()
        message = msg.get("message", "")
        if level == "INSIGHT":
             logger.info(f"INSIGHT: {message}")
        else:
             logger.info(f"ENGINE: {message}")


    # --- Connect signals ---
    orchestrator.state_changed.connect(on_state_changed)
    orchestrator.log_message.connect(on_log_message)

    # --- Dispatch actions to configure and run the experiment ---
    logger.info(f"Setting challenge to: {challenge_config['name']}")
    orchestrator.dispatch(ActionType.SET_CHALLENGE, challenge_config)

    for algo in algorithms_to_run:
        logger.info(f"Adding algorithm: {algo['name']}")
        orchestrator.dispatch(ActionType.ADD_ALGORITHM, algo)

    logger.info("Starting experiment run...")
    orchestrator.dispatch(ActionType.START_RUN, execution_settings)

    # --- Wait for experiment to finish ---
    experiment_done.wait()
    end_time = time.time()

    # --- Print final results ---
    final_experiment_state = orchestrator.experiment
    logger.info(
        f"\n--- Experiment Finished in {end_time - start_time:.2f} seconds ---"
    )
    for trial in final_experiment_state.trials.values():
        results_str = "".join(
            [f"\n    {m}: {v}" for m, v in trial.results.items()]
        )
        logger.info(
            f"\nTrial ID: {trial.id}"
            f"\n  Algorithm: {trial.algorithm_name}"
            f"\n  Status: {trial.status.value}"
            f"\n  Hyperparameters: {trial.hyperparameters}"
            f"\n  Results: {results_str}"
        )

    orchestrator.shutdown()


if __name__ == "__main__":
    main()
