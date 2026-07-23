"""
test_utils.py

Unit tests for src.utils.
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    create_directory,
    load_object,
    print_dataframe_info,
    print_metrics,
    print_separator,
    save_object,
    set_random_seed,
    setup_logger,
)


# ============================================================
# Fixtures
#
# NOTE: temp_dir is defined here so this file is runnable on its own.
# If test_preprocess.py / test_model.py / test_api.py also need a
# temp directory, move this fixture into tests/conftest.py instead so
# it's shared rather than duplicated across test files.
# ============================================================

@pytest.fixture
def temp_dir(tmp_path):
    """
    Provide a per-test temporary directory (auto-cleaned by pytest).
    """
    return tmp_path


# ============================================================
# Logger
# ============================================================

def test_setup_logger():
    """Logger should be created correctly."""

    logger = setup_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert len(logger.handlers) >= 2


def test_setup_logger_no_duplicate_handlers():
    """
    Calling setup_logger twice with the same name should return the
    same logger instance without stacking duplicate handlers. This
    guards the `if logger.handlers: return logger` check in utils.py —
    without it, repeated imports/calls would keep adding handlers and
    every log line would print multiple times.
    """

    logger_first = setup_logger("duplicate_test_logger")
    handler_count_after_first = len(logger_first.handlers)

    logger_second = setup_logger("duplicate_test_logger")

    assert logger_first is logger_second
    assert len(logger_second.handlers) == handler_count_after_first


# ============================================================
# Directory Utilities
# ============================================================

def test_create_directory(temp_dir):
    """Directory should be created if it does not exist."""

    new_dir = temp_dir / "new_folder"

    create_directory(new_dir)

    assert new_dir.exists()
    assert new_dir.is_dir()


def test_create_directory_is_idempotent(temp_dir):
    """
    Calling create_directory twice on the same path should not raise,
    since it relies on exist_ok=True.
    """

    new_dir = temp_dir / "repeat_folder"

    create_directory(new_dir)
    create_directory(new_dir)  # should not raise

    assert new_dir.exists()


# ============================================================
# Joblib Utilities
# ============================================================

def test_save_and_load_object(temp_dir):
    """Saved object should be identical after loading."""

    obj = {
        "name": "Credit Card Fraud",
        "version": 1,
        "accuracy": 0.99,
    }

    file_path = temp_dir / "object.joblib"

    save_object(obj, file_path)

    assert file_path.exists()

    loaded = load_object(file_path)

    assert loaded == obj


def test_save_object_creates_parent_directories(temp_dir):
    """
    save_object should create any missing parent directories rather
    than raising, since callers (e.g. preprocess.py saving the scaler)
    rely on this behavior.
    """

    nested_path = temp_dir / "nested" / "sub" / "scaler.joblib"

    save_object({"fitted": True}, nested_path)

    assert nested_path.exists()


def test_load_object_file_not_found(temp_dir):
    """Loading a missing object should raise FileNotFoundError."""

    missing_file = temp_dir / "missing.joblib"

    with pytest.raises(FileNotFoundError):
        load_object(missing_file)


# ============================================================
# Random Seed
# ============================================================

def test_set_random_seed():
    """Random seed should produce reproducible results."""

    set_random_seed(42)

    python_random = random.random()
    numpy_random = np.random.rand()

    set_random_seed(42)

    assert random.random() == python_random
    assert np.random.rand() == numpy_random


# ============================================================
# Print Utilities
# ============================================================

def test_print_separator(capsys):
    """Separator should print '=' characters."""

    print_separator()

    captured = capsys.readouterr()

    assert "=" in captured.out
    assert len(captured.out.strip()) == 70


def test_print_dataframe_info(capsys):
    """DataFrame information should be printed."""

    df = pd.DataFrame(
        {
            "A": [1, 2, None],
            "B": [4, 5, 6],
        }
    )

    print_dataframe_info(df)

    captured = capsys.readouterr()

    assert "Shape" in captured.out
    assert "Missing Values" in captured.out
    assert "Duplicate Rows" in captured.out


def test_print_metrics(capsys):
    """Numeric metrics should be formatted to 4 decimal places."""

    metrics = {
        "accuracy": 0.98,
        "precision": 0.91,
        "recall": 0.95,
    }

    print_metrics(metrics)

    captured = capsys.readouterr()

    assert "Model Performance" in captured.out
    assert "0.9800" in captured.out
    assert "0.9100" in captured.out
    assert "0.9500" in captured.out


def test_print_metrics_handles_non_numeric_values(capsys):
    """
    Non-numeric values (e.g. a model name string) should be printed
    as-is rather than crashing on the `.4f` format spec. This covers
    the defensive isinstance check added to print_metrics.
    """

    metrics = {
        "accuracy": 0.98,
        "model": "RandomForest",
    }

    print_metrics(metrics)  # should not raise

    captured = capsys.readouterr()

    assert "RandomForest" in captured.out
    assert "0.9800" in captured.out


# ============================================================
# Path Handling
# ============================================================

def test_save_object_accepts_string_path(temp_dir):
    """save_object should accept string paths."""

    obj = [1, 2, 3]

    file_path = str(temp_dir / "list.joblib")

    save_object(obj, file_path)

    loaded = load_object(file_path)

    assert loaded == obj


def test_create_directory_accepts_string_path(temp_dir):
    """create_directory should accept string paths."""

    path = str(temp_dir / "string_dir")

    create_directory(path)

    assert Path(path).exists()
    assert Path(path).is_dir()