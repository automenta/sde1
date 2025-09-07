import os

"""
Centralized configuration for the Scientific Discovery Engine (SDE).

This module contains default values and settings that are used across the
application. Centralizing them here makes the system easier to configure
and understand.
"""

# --- Runtime Engine Settings ---

# The default directory for storing model checkpoints.
CHECKPOINTS_DIR = "./checkpoints"


# --- Initialization ---

# Ensure the checkpoints directory exists.
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
