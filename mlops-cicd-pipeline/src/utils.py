"""
utils.py

Utility functions for the Credit Card Fraud Detection
MLOps Pipeline.

Contains reusable helper functions for:
- Logging
- Directory creation
- Saving/loading objects
- Random seed initialization
"""

import logging
import random
from pathlib import Path

import joblib
import numpy as np

from src.config import LOG_FILE, LOG_LEVEL


# ============================================================
# Logger
# ============================================================

def setup_logger(name: str = "mlops") -> logging.Logger:
    """
    Configure and return a logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Ensure the log directory exists before attaching the file handler,
    # even if config.py's directory-creation loop hasn't run yet for
    # some reason (e.g. utils imported in isolation during testing).
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# ============================================================
# Directory Utilities
# ============================================================

def create_directory(path: str | Path) -> None:
    """
    Create directory if it does not exist.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# Joblib Utilities
# ============================================================

def save_object(obj, filepath: str | Path) -> None:
    """
    Save a Python object using joblib.

    Parameters
    ----------
    obj : Any
        Object to save.

    filepath : str | Path
        Destination file.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(obj, filepath)

    logger.info("Object saved to %s", filepath)


def load_object(filepath: str | Path):
    """
    Load a saved joblib object.

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    Any
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist.")

    logger.info("Loading object from %s", filepath)

    return joblib.load(filepath)


# ============================================================
# Random Seed
# ============================================================

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    logger.info("Random seed set to %d", seed)


# ============================================================
# Data Information
# ============================================================

def print_separator(length: int = 70) -> None:
    """
    Print a separator line.
    """

    print("=" * length)


def print_dataframe_info(df) -> None:
    """
    Display useful dataframe information.

    Parameters
    ----------
    df : pandas.DataFrame
    """

    print_separator()

    print("Shape")

    print(df.shape)

    print_separator()

    print("Missing Values")

    print(df.isnull().sum())

    print_separator()

    print("Duplicate Rows")

    print(df.duplicated().sum())

    print_separator()


# ============================================================
# Model Information
# ============================================================

def print_metrics(metrics: dict) -> None:
    """
    Pretty-print evaluation metrics.

    Parameters
    ----------
    metrics : dict
    """

    print_separator()

    print("Model Performance")

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key:<15}: {value:.4f}")
        else:
            print(f"{key:<15}: {value}")

    print_separator()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    logger.info("Testing utility functions...")

    set_random_seed()

    print_separator()

    print("Utilities loaded successfully.")

    print_separator()