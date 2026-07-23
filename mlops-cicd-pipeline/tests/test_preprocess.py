"""
test_preprocess.py

Unit tests for src.preprocess.
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src import preprocess
from src.config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    SCALER_TYPE,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)

SCALER_MAPPING = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


# ============================================================
# Fixtures
#
# NOTE: sample_dataframe is used across most tests in this file and
# will likely be needed by test_model.py too. Consider moving it into
# tests/conftest.py so both files share one definition instead of
# duplicating it.
# ============================================================

@pytest.fixture
def sample_dataframe():
    """
    Create a small balanced dataset suitable for stratified splitting.
    """
    rows = 100

    data = {
        "Time": list(range(rows)),
        "Amount": [float(i) for i in range(rows)],
    }

    # Dummy PCA features
    for i in range(1, 29):
        data[f"V{i}"] = [float(i)] * rows

    # 50 genuine, 50 fraud
    data[TARGET_COLUMN] = [0] * 50 + [1] * 50

    return pd.DataFrame(data)


# ============================================================
# Load
# ============================================================

def test_load_raw_data_file_not_found(monkeypatch):
    """Missing raw dataset should raise FileNotFoundError."""

    monkeypatch.setattr(preprocess, "RAW_DATA_PATH", Path("does_not_exist.csv"))

    with pytest.raises(FileNotFoundError):
        preprocess.load_raw_data()


# ============================================================
# Validation
# ============================================================

def test_validate_columns_success(sample_dataframe):
    """Should not raise when target and all feature columns are present."""

    preprocess.validate_columns(sample_dataframe)  # no exception = pass


def test_validate_columns_missing_target(sample_dataframe):
    df = sample_dataframe.drop(columns=[TARGET_COLUMN])

    with pytest.raises(KeyError):
        preprocess.validate_columns(df)


def test_validate_columns_missing_feature(sample_dataframe):
    df = sample_dataframe.drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(KeyError):
        preprocess.validate_columns(df)


# ============================================================
# Duplicate Removal
# ============================================================

def test_drop_duplicate_rows_enabled(monkeypatch, sample_dataframe):
    duplicated = pd.concat(
        [sample_dataframe, sample_dataframe.iloc[[0]]],
        ignore_index=True,
    )

    monkeypatch.setattr(preprocess, "DROP_DUPLICATES", True)

    cleaned = preprocess.drop_duplicate_rows(duplicated)

    assert len(cleaned) == len(sample_dataframe)


def test_drop_duplicate_rows_disabled(monkeypatch, sample_dataframe):
    duplicated = pd.concat(
        [sample_dataframe, sample_dataframe.iloc[[0]]],
        ignore_index=True,
    )

    monkeypatch.setattr(preprocess, "DROP_DUPLICATES", False)

    cleaned = preprocess.drop_duplicate_rows(duplicated)

    assert len(cleaned) == len(duplicated)


# ============================================================
# Split
# ============================================================

def test_split_data(sample_dataframe):
    train_df, val_df, test_df = preprocess.split_data(sample_dataframe)

    assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataframe)

    expected_test = int(len(sample_dataframe) * TEST_SIZE)
    expected_val = int(len(sample_dataframe) * VALIDATION_SIZE)

    assert abs(len(test_df) - expected_test) <= 1
    assert abs(len(val_df) - expected_val) <= 1

    # ensure stratification approximately preserved
    assert train_df[TARGET_COLUMN].mean() == pytest.approx(0.5, rel=0.2)
    assert val_df[TARGET_COLUMN].mean() == pytest.approx(0.5, rel=0.2)
    assert test_df[TARGET_COLUMN].mean() == pytest.approx(0.5, rel=0.2)


def test_split_data_is_reproducible(sample_dataframe):
    """
    Two calls to split_data on the same input, under the same
    RANDOM_STATE, should produce identical splits. This is the whole
    point of setting RANDOM_STATE in config.py — if this ever breaks,
    every downstream metric becomes non-reproducible.
    """

    train_1, val_1, test_1 = preprocess.split_data(sample_dataframe)
    train_2, val_2, test_2 = preprocess.split_data(sample_dataframe)

    pd.testing.assert_frame_equal(train_1, train_2)
    pd.testing.assert_frame_equal(val_1, val_2)
    pd.testing.assert_frame_equal(test_1, test_2)


def test_split_data_no_overlap(sample_dataframe):
    """
    Train/val/test splits should not share any rows (checked via
    index, since sample_dataframe has a unique row per index here).
    """

    train_df, val_df, test_df = preprocess.split_data(sample_dataframe)

    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


# ============================================================
# Scaler
# ============================================================

def test_get_scaler():
    scaler = preprocess.get_scaler()

    assert isinstance(scaler, SCALER_MAPPING[SCALER_TYPE])


@pytest.mark.parametrize("scaler_type", list(SCALER_MAPPING.keys()))
def test_get_scaler_all_supported_types(monkeypatch, scaler_type):
    """
    Exercise every branch of the scaler registry, not just whichever
    SCALER_TYPE happens to be set in config right now — otherwise two
    of the three branches are never actually tested.
    """

    monkeypatch.setattr(preprocess, "SCALER_TYPE", scaler_type)

    scaler = preprocess.get_scaler()

    assert isinstance(scaler, SCALER_MAPPING[scaler_type])


def test_get_scaler_invalid(monkeypatch):
    monkeypatch.setattr(preprocess, "SCALER_TYPE", "InvalidScaler")

    with pytest.raises(ValueError):
        preprocess.get_scaler()


# ============================================================
# Scaling
# ============================================================

def test_scale_features(monkeypatch, sample_dataframe, tmp_path):
    train_df, val_df, test_df = preprocess.split_data(sample_dataframe)

    scaler_path = tmp_path / "scaler.joblib"

    monkeypatch.setattr(preprocess, "SCALER_PATH", scaler_path)

    train_scaled, val_scaled, test_scaled = preprocess.scale_features(
        train_df,
        val_df,
        test_df,
    )

    assert scaler_path.exists()

    if preprocess.SCALE_AMOUNT:
        assert train_scaled["Amount"].equals(train_df["Amount"]) is False

    if preprocess.SCALE_TIME:
        assert train_scaled["Time"].equals(train_df["Time"]) is False


def test_scale_features_no_scaling(monkeypatch, sample_dataframe):
    train_df, val_df, test_df = preprocess.split_data(sample_dataframe)

    monkeypatch.setattr(preprocess, "SCALE_AMOUNT", False)
    monkeypatch.setattr(preprocess, "SCALE_TIME", False)

    train_scaled, val_scaled, test_scaled = preprocess.scale_features(
        train_df,
        val_df,
        test_df,
    )

    pd.testing.assert_frame_equal(train_df, train_scaled)
    pd.testing.assert_frame_equal(val_df, val_scaled)
    pd.testing.assert_frame_equal(test_df, test_scaled)


def test_scale_features_no_leakage(monkeypatch, sample_dataframe, tmp_path):
    """
    The scaler must be fit on the train split only. This test manually
    computes what StandardScaler *would* produce if fit on train alone,
    and confirms scale_features matches that — catching the common bug
    of accidentally fitting on train+val+test combined (which would
    leak test-set statistics into the transform).
    """

    monkeypatch.setattr(preprocess, "SCALER_TYPE", "StandardScaler")
    monkeypatch.setattr(preprocess, "SCALE_AMOUNT", True)
    monkeypatch.setattr(preprocess, "SCALE_TIME", False)
    monkeypatch.setattr(preprocess, "SCALER_PATH", tmp_path / "scaler.joblib")

    train_df, val_df, test_df = preprocess.split_data(sample_dataframe)

    _, val_scaled, _ = preprocess.scale_features(train_df, val_df, test_df)

    expected_scaler = StandardScaler()
    expected_scaler.fit(train_df[["Amount"]])
    expected_val_amount = expected_scaler.transform(val_df[["Amount"]]).flatten()

    assert val_scaled["Amount"].to_numpy() == pytest.approx(expected_val_amount)


# ============================================================
# Save
# ============================================================

def test_save_processed_splits(monkeypatch, sample_dataframe, tmp_path):
    train_df, val_df, test_df = preprocess.split_data(sample_dataframe)

    monkeypatch.setattr(preprocess, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(preprocess, "TRAIN_DATA_PATH", tmp_path / "train.csv")
    monkeypatch.setattr(preprocess, "VAL_DATA_PATH", tmp_path / "val.csv")
    monkeypatch.setattr(preprocess, "TEST_DATA_PATH", tmp_path / "test.csv")

    preprocess.save_processed_splits(train_df, val_df, test_df)

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()


# ============================================================
# Pipeline
# ============================================================

def test_run_preprocessing(monkeypatch):
    """
    Verify the orchestration calls each stage exactly once, in order.
    """

    calls = []

    dummy_df = pd.DataFrame()

    monkeypatch.setattr(preprocess, "load_raw_data", lambda: dummy_df)
    monkeypatch.setattr(preprocess, "print_dataframe_info", lambda df: calls.append("info"))
    monkeypatch.setattr(preprocess, "validate_columns", lambda df: calls.append("validate"))
    monkeypatch.setattr(
        preprocess,
        "drop_duplicate_rows",
        lambda df: (calls.append("drop_duplicates"), df)[1],
    )
    monkeypatch.setattr(
        preprocess,
        "split_data",
        lambda df: (calls.append("split"), (dummy_df, dummy_df, dummy_df))[1],
    )
    monkeypatch.setattr(
        preprocess,
        "scale_features",
        lambda a, b, c: (calls.append("scale"), (a, b, c))[1],
    )
    monkeypatch.setattr(
        preprocess,
        "save_processed_splits",
        lambda a, b, c: calls.append("save"),
    )

    preprocess.run_preprocessing()

    assert calls == ["info", "validate", "drop_duplicates", "split", "scale", "save"]