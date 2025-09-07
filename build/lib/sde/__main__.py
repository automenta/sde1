import logging
import time
import uuid

from sde.core.types import Trial
from sde.engine.orchestrator import Orchestrator
from sde.exploration.schedulers import SuccessiveHalvingScheduler


def main():
    """Main entry point for running a command-line-based SDE experiment.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # --- Experiment Configuration ---
    trials_to_run = [
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="SimpleCNN",
            hyperparameters={
                "model_params": {"dropout_rate": 0.25},
                "optimizer_params": {"name": "Adam", "lr": 0.01},
            },
        ),
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="SimpleCNN",
            hyperparameters={
                "model_params": {"dropout_rate": 0.5},
                "optimizer_params": {"name": "Adam", "lr": 0.001},
            },
        ),
        Trial(
            id=f"trial_{uuid.uuid4().hex[:6]}",
            algorithm_name="LogisticRegression",
            hyperparameters={"optimizer_params": {"name": "SGD", "lr": 0.01}},
        ),
    ]
    DATASET = "MNIST"
    # --- End Configuration ---

    logger.info(f"Starting experiment with {len(trials_to_run)} trials on {DATASET}.")

    adaptive_scheduler = SuccessiveHalvingScheduler(
        metric="accuracy", increasing=True, min_epochs_per_rung=2
    )

    orchestrator = Orchestrator(
        trials=trials_to_run,
        dataset_name=DATASET,
        adaptive_scheduler=adaptive_scheduler,
        max_workers=2,
        enable_checkpointing=True,
    )

    # --- Connect signals to loggers ---
    orchestrator.log_message.connect(lambda msg: logger.info(f"ORCHESTRATOR: {msg}"))
    orchestrator.insight_generated.connect(lambda msg: logger.info(f"INSIGHT: {msg}"))
    orchestrator.trial_updated.connect(
        lambda data: logger.info(
            f"TRIAL UPDATE: {data['id']} - Status: {data['status']}"
        )
    )

    start_time = time.time()
    main_thread = orchestrator.start()
    main_thread.join()  # Wait for the experiment to complete
    end_time = time.time()

    logger.info(f"\n--- Experiment Finished in {end_time - start_time:.2f} seconds ---")
    final_trials = orchestrator.datastore.get_all_trials()
    for trial in final_trials.values():
        results_str = "".join([f"\n    {m}: {v}" for m, v in trial.results.items()])
        logger.info(
            f"\nTrial ID: {trial.id}"
            f"\n  Algorithm: {trial.algorithm_name}"
            f"\n  Status: {trial.status.value}"
            f"\n  Hyperparameters: {trial.hyperparameters}"
            f"\n  Results: {results_str}"
        )


if __name__ == "__main__":
    main()
