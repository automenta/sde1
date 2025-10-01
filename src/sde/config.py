import atexit
import tempfile
from pathlib import Path
from typing import Callable

from platformdirs import PlatformDirs

# --- Application Metadata ---
APP_NAME = "sde"
APP_AUTHOR = "SDE"


# --- Demo Mode ---
# When True, the application runs a scripted demonstration.
DEMO_MODE = False


# --- Auto Mode ---
# When True, the application runs in a continuous, automated discovery mode.
AUTO_MODE = False


_temp_dir_obj = None


def _get_temp_dir_path() -> Path:
    """Returns path to a temp dir, creating it if it doesn't exist."""
    global _temp_dir_obj
    if _temp_dir_obj is None:
        # Create a temporary directory that will be cleaned up automatically on exit
        _temp_dir_obj = tempfile.TemporaryDirectory()
        atexit.register(_temp_dir_obj.cleanup)
    return Path(_temp_dir_obj.name)


# --- Platform-specific Paths ---
_dirs = PlatformDirs(appname=APP_NAME, appauthor=False)


# --- Default Values for UI and Engine ---
class Defaults:
    """A namespace for default configuration values."""

    # For setup_pane.py
    FALLBACK_MAX_WORKERS = 4
    DEFAULT_WORK_UNIT_TIMEOUT_S = 300
    MAX_WORK_UNIT_TIMEOUT_S = 3600  # 1 hour
    DEFAULT_TRIALS_PER_ALGO = 10
    MAX_TRIALS_PER_ALGO = 1000

    # For adaptive schedulers
    DEFAULT_ADAPTIVE_POLICY = "SuccessiveHalving"


# --- Auto Mode Configuration ---
class AutoModeConfig:
    """A namespace for auto mode configuration values."""

    # Time in seconds to wait before starting a new experiment
    DELAY_BETWEEN_EXPERIMENTS_S = 10
    # Default patience budget for each automated run
    PATIENCE_BUDGET = {"max_wall_clock_time_s": 120}


# --- Public API ---


def _get_and_create_dir(path_func: Callable[[], Path], *subdirs: str) -> Path:
    """A helper to get a base directory, append subdirectories, and ensure it exists."""
    base_path = path_func()
    # Using the *subdirs syntax, this becomes "base_path / sub1 / sub2"
    final_path = base_path.joinpath(*subdirs)
    final_path.mkdir(parents=True, exist_ok=True)
    return final_path


def get_user_data_dir() -> Path:
    """Returns the platform-specific user data directory for the application.
    This is where persistent data like experiments and checkpoints should be stored.
    In demo mode, this will be a temporary directory.
    """
    if DEMO_MODE:
        return _get_and_create_dir(_get_temp_dir_path)
    return _get_and_create_dir(lambda: _dirs.user_data_path)


def get_user_cache_dir() -> Path:
    """Returns the platform-specific user cache directory for the application.
    This is where non-essential, downloadable data like datasets should be stored.
    In demo mode, this will be a temporary directory.
    """
    if DEMO_MODE:
        return _get_and_create_dir(_get_temp_dir_path)
    return _get_and_create_dir(lambda: _dirs.user_cache_path)


def get_checkpoints_dir() -> Path:
    """Returns the specific directory for storing model checkpoints."""
    if DEMO_MODE:
        return _get_and_create_dir(_get_temp_dir_path, "checkpoints")
    return _get_and_create_dir(get_user_data_dir, "checkpoints")
