from typing import Dict

from .core.definitions import DatasetDefinition
from .core.definitions import ModelDefinition
from .exploration.schedulers import AdaptiveScheduler


class Registry:
    """A central registry for all discoverable components like challenges, models, and schedulers."""

    def __init__(self):
        self.challenges: Dict[str, DatasetDefinition] = {}
        self.models: Dict[str, ModelDefinition] = {}
        self.schedulers: Dict[str, type[AdaptiveScheduler]] = {}

    def register_challenge(self, challenge: DatasetDefinition):
        """Register a new challenge."""
        if challenge.name in self.challenges:
            # For now, we can just warn. In a real app, might want to raise an error.
            print(
                f"Warning: Challenge '{challenge.name}' is already registered. Overwriting."
            )
        self.challenges[challenge.name] = challenge

    def register_model(self, model: ModelDefinition):
        """Register a new model."""
        if model.name in self.models:
            print(f"Warning: Model '{model.name}' is already registered. Overwriting.")
        self.models[model.name] = model

    def register_scheduler(self, name: str, scheduler_class: type[AdaptiveScheduler]):
        """Register a new scheduler."""
        if name in self.schedulers:
            print(f"Warning: Scheduler '{name}' is already registered. Overwriting.")
        self.schedulers[name] = scheduler_class

    def get_challenge(self, name: str) -> DatasetDefinition:
        return self.challenges[name]

    def get_model(self, name: str) -> ModelDefinition:
        return self.models[name]

    def get_scheduler_class(self, name: str) -> type[AdaptiveScheduler]:
        return self.schedulers[name]

    def list_challenges(self):
        return list(self.challenges.keys())

    def list_models(self):
        return list(self.models.keys())

    def list_schedulers(self):
        return list(self.schedulers.keys())


# Create a global instance of the registry
registry = Registry()
