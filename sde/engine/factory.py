import logging
from typing import Dict, Any

from sde.exploration.schedulers import (
    AdaptiveScheduler,
    SuccessiveHalvingScheduler,
    HyperbandScheduler,
)
from sde.challenges import AVAILABLE_DATASETS

logger = logging.getLogger(__name__)

SCHEDULER_MAP = {
    "SuccessiveHalving": SuccessiveHalvingScheduler,
    "Hyperband": HyperbandScheduler,
}


class SchedulerFactory:
    """A factory for creating adaptive scheduler instances."""

    @staticmethod
    def create_scheduler(
        policy_name: str,
        challenge_name: str,
        patience_budget: Dict[str, Any] = None,
    ) -> AdaptiveScheduler:
        """
        Creates an instance of an adaptive scheduler based on its name and config.

        Args:
            policy_name: The name of the scheduling policy (e.g., "Hyperband").
            challenge_name: The name of the challenge to get metric info.
            patience_budget: The experiment's patience budget, which may contain
                             scheduler-specific settings.

        Returns:
            An instance of the specified AdaptiveScheduler.

        Raises:
            ValueError: If the policy_name is unknown or required config is missing.
        """
        if patience_budget is None:
            patience_budget = {}

        scheduler_class = SCHEDULER_MAP.get(policy_name)
        if not scheduler_class:
            raise ValueError(f"Unknown scheduler '{policy_name}' specified.")

        try:
            challenge_def = AVAILABLE_DATASETS[challenge_name]
            metric = challenge_def.performance_metric_name
            increasing = "accuracy" in metric.lower()
        except KeyError:
            raise ValueError(f"Unknown challenge name '{challenge_name}' provided.")

        # --- Instantiate the scheduler with correct parameters ---
        scheduler_args = {
            "metric": metric,
            "increasing": increasing,
        }

        if policy_name == "Hyperband":
            # Hyperband requires max_resource_per_trial.
            # This logic is now centralized here.
            max_resource = patience_budget.get("max_epochs", 81)  # Default value
            scheduler_args["max_resource_per_trial"] = max_resource
            logger.info(
                f"Instantiating Hyperband with max_resource_per_trial={max_resource}"
            )

        # Future schedulers with special requirements can be added here.

        logger.info(
            f"Creating '{policy_name}' scheduler instance with args: {scheduler_args}"
        )
        return scheduler_class(**scheduler_args)
