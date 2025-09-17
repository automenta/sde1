from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Union

from .core.definitions import DatasetDefinition
from .core.definitions import ModelDefinition
from .exploration.schedulers import AdaptiveScheduler


@dataclass(frozen=True)
class RegisteredComponent:
    """A container for a registered component and its metadata."""

    component: Union[DatasetDefinition, ModelDefinition, type[AdaptiveScheduler]]
    version: str
    origin_file: str
    dependencies: Dict[str, str] = field(default_factory=dict)


class Registry:
    """A central registry for all discoverable components."""

    def __init__(self, strict_mode: bool = False):
        self.challenges: Dict[str, RegisteredComponent] = {}
        self.models: Dict[str, RegisteredComponent] = {}
        self.schedulers: Dict[str, RegisteredComponent] = {}
        self.strict_mode = strict_mode

    def _register(
        self,
        registry_dict: Dict[str, RegisteredComponent],
        component: Union[DatasetDefinition, ModelDefinition, type[AdaptiveScheduler]],
        name: str,
        version: str,
        origin_file: str,
        dependencies: Dict[str, str],
    ):
        """Helper method to register a component."""
        if name in registry_dict:
            message = f"Warning: Component '{name}' is already registered. Overwriting."
            print(message)
            if self.strict_mode:
                raise ValueError(message)
        registry_dict[name] = RegisteredComponent(
            component=component,
            version=version,
            origin_file=origin_file,
            dependencies=dependencies,
        )

    def register_challenge(
        self,
        challenge: DatasetDefinition,
        origin_file: str,
    ):
        self._register(
            self.challenges,
            challenge,
            challenge.name,
            challenge.version,
            origin_file,
            challenge.dependencies,
        )

    def register_model(self, model: ModelDefinition, origin_file: str):
        self._register(
            self.models,
            model,
            model.name,
            model.version,
            origin_file,
            model.dependencies,
        )

    def register_scheduler(
        self, scheduler_class: type[AdaptiveScheduler], origin_file: str
    ):
        # Schedulers don't have a formal definition class, so we get metadata from attrs
        name = getattr(scheduler_class, "NAME", scheduler_class.__name__)
        version = getattr(scheduler_class, "VERSION", "0.1.0")
        dependencies = getattr(scheduler_class, "DEPENDENCIES", {})
        self._register(
            self.schedulers, scheduler_class, name, version, origin_file, dependencies
        )

    def get_challenge(self, name: str) -> DatasetDefinition:
        return self.challenges[name].component

    def get_model(self, name: str) -> ModelDefinition:
        return self.models[name].component

    def get_scheduler_class(self, name: str) -> type[AdaptiveScheduler]:
        return self.schedulers[name].component

    def list_challenges(self) -> list[str]:
        return list(self.challenges.keys())

    def list_models(self) -> list[str]:
        return list(self.models.keys())

    def list_schedulers(self) -> list[str]:
        return list(self.schedulers.keys())


# Create a global instance of the registry
registry = Registry()
