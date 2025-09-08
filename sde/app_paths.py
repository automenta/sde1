from pathlib import Path

from platformdirs import PlatformDirs

# Define app-specific directory names
_dirs = PlatformDirs(appname="sde", appauthor=False)

# --- Public API ---

def get_user_data_dir() -> Path:
    """Returns the platform-specific user data directory for the application.
    This is where persistent data like experiments and checkpoints should be stored.
    The directory is created if it does not exist.
    """
    path = _dirs.user_data_path
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_user_cache_dir() -> Path:
    """Returns the platform-specific user cache directory for the application.
    This is where non-essential, downloadable data like datasets should be stored.
    The directory is created if it does not exist.
    """
    path = _dirs.user_cache_path
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_checkpoints_dir() -> Path:
    """Returns the specific directory for storing model checkpoints.
    The directory is created if it does not exist.
    """
    path = get_user_data_dir() / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path
