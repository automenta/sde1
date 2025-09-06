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
# This should typically be set to the number of available CPU cores.
MAX_WORKERS = 2

# The default directory for storing model checkpoints.
CHECKPOINTS_DIR = "./checkpoints"
