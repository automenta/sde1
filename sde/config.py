import os

"""
Centralized configuration for the Scientific Discovery Engine (SDE).

This module contains default values and settings that are used across the
application. Centralizing them here makes the system easier to configure
and understand.
"""

# --- Experiment Settings ---

# The default number of trials to generate for each algorithm.
NUM_TRIALS_PER_ALGO = 10


# --- Runtime Engine Settings ---

# The default number of parallel worker processes to run.
# We default to the number of available CPU cores, falling back to 2 if it's not determinable.
MAX_WORKERS = os.cpu_count() or 2

# The default directory for storing model checkpoints.
CHECKPOINTS_DIR = "./checkpoints"
