import importlib
import inspect
import logging
import pkgutil

from .core.definitions import DatasetDefinition
from .core.definitions import ModelDefinition
from .exploration.schedulers import AdaptiveScheduler
from .registry import registry

logger = logging.getLogger(__name__)

# --- Component Base Names ---
# We'll look for variables with these names in the discovered modules.
CHALLENGE_VAR_NAME = "CHALLENGE"
MODEL_VAR_NAME = "MODEL"
SCHEDULER_VAR_NAME = "SCHEDULER"


def discover_and_register_components():
    """Discover and register all pluggable components.

    This function serves as the main entry point for the component discovery
    process. It scans predefined packages (challenges, models, schedulers) for
    Python modules, imports them, and registers any components they contain with
    the central registry.

    The discovery is convention-based, meaning it looks for variables with
    specific names (e.g., `CHALLENGE`, `MODEL`) or classes that inherit from
    specific base classes (e.g., `AdaptiveScheduler`).
    """
    logger.info("Starting component discovery...")

    # --- Discover Challenges ---
    import sde.challenges

    _discover_in_package(sde.challenges, DatasetDefinition, registry.register_challenge)

    # --- Discover Models ---
    import sde.models

    _discover_in_package(sde.models, ModelDefinition, registry.register_model)

    # --- Discover Schedulers ---
    import sde.exploration

    _discover_schedulers(sde.exploration)

    logger.info("Discovery complete. Found:")
    logger.info(f"  - Challenges: {registry.list_challenges()}")
    logger.info(f"  - Models: {registry.list_models()}")
    logger.info(f"  - Schedulers: {registry.list_schedulers()}")


def _discover_in_package(package, base_class, register_func):
    """Find and register components in a given package."""
    for _, name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            module = importlib.import_module(name)
            if not ispkg:
                # Find all variables of the specified base class type
                for var_name, var_value in module.__dict__.items():
                    if isinstance(var_value, base_class):
                        logger.debug(
                            f"Found {base_class.__name__} '{var_value.name}' in {name}"
                        )
                        register_func(var_value, origin_file=module.__file__)
        except Exception as e:
            logger.warning(f"Could not import or register from module {name}: {e}")


def _discover_schedulers(package):
    """Specific helper for schedulers as they are classes, not instances."""
    for _, name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if ispkg:
            continue
        try:
            module = importlib.import_module(name)
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(class_obj, AdaptiveScheduler)
                    and class_obj is not AdaptiveScheduler
                ):
                    # Use a readable name for the scheduler, e.g., "Successive Halving"
                    logger.debug(f"Found Scheduler '{class_name}' in {name}")
                    registry.register_scheduler(
                        class_obj, origin_file=module.__file__
                    )
        except Exception as e:
            logger.warning(
                f"Could not import or register scheduler from module {name}: {e}"
            )
