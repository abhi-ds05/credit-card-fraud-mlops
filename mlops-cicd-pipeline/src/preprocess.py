"""
preprocess.py

Data preprocessing pipeline for the Credit Card Fraud Detection
MLOps project.

Responsibilities:
- Load raw data
- Drop duplicates (if configured)
- Split into train / validation / test sets (stratified on Class)
- Scale the `Amount` feature (fit on train only, applied to all splits)
- Persist processed splits to data/processed/
- Persist the fitted scaler to models/scaler.joblib

This script is meant to be run standalone:
    python src/preprocess.py
"""

import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    STRATIFY,
    DROP_DUPLICATES,
    SCALE_AMOUNT,
    SCALE_TIME,
    SCALER_TYPE,
)
from src.utils import logger, save_object, print_dataframe_info, set_random_seed

# Validation split is not defined in config.py's path constants yet —
# derived here for consistency with TRAIN_DATA_PATH / TEST_DATA_PATH.
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"

SCALER_REGISTRY = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


# ============================================================
# Load
# ============================================================

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw credit card transactions dataset.
    """
    if not RAW_DATA_PATH.exists():
        logger.error("Raw data file not found at %s", RAW_DATA_PATH)
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. "
            "Download creditcard.csv and place it there before running preprocessing."
        )

    logger.info("Loading raw data from %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info("Loaded raw data with shape %s", df.shape)

    return df


# ============================================================
# Clean
# ============================================================

def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows if configured to do so.
    """
    if not DROP_DUPLICATES:
        logger.info("DROP_DUPLICATES is False — skipping duplicate removal.")
        return df

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    logger.info("Dropped %d duplicate rows (%.4f%% of data).", removed, (removed / before) * 100)

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Confirm the target column and expected feature columns are present
    before doing any further processing.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise KeyError(f"Expected feature columns missing from dataset: {missing_features}")

    logger.info("Column validation passed: target and all expected features present.")


# ============================================================
# Split
# ============================================================

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train / validation / test sets.

    Splitting is stratified on the target column by default (STRATIFY),
    which matters a lot here since fraud is ~0.17% of the data — an
    unstratified split risks a test set with too few (or zero) fraud
    cases to evaluate against.
    """
    stratify_col = df[TARGET_COLUMN] if STRATIFY else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_col,
    )

    # VALIDATION_SIZE is expressed as a fraction of the *original* dataset,
    # so it needs to be rescaled relative to what remains after the test split.
    relative_val_size = VALIDATION_SIZE / (1 - TEST_SIZE)

    stratify_col_train_val = train_val_df[TARGET_COLUMN] if STRATIFY else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=stratify_col_train_val,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Fraud counts — train: %d, val: %d, test: %d",
        train_df[TARGET_COLUMN].sum(),
        val_df[TARGET_COLUMN].sum(),
        test_df[TARGET_COLUMN].sum(),
    )

    return train_df, val_df, test_df


# ============================================================
# Scale
# ============================================================

def get_scaler():
    """
    Instantiate the scaler class configured in config.SCALER_TYPE.
    """
    if SCALER_TYPE not in SCALER_REGISTRY:
        raise ValueError(
            f"Unsupported SCALER_TYPE '{SCALER_TYPE}'. "
            f"Supported options: {list(SCALER_REGISTRY.keys())}"
        )

    return SCALER_REGISTRY[SCALER_TYPE]()


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit the configured scaler on the train split only, then apply it to
    train/val/test. Fitting on train-only avoids leaking information
    from validation/test data into the scaling parameters.

    Only `Amount` is scaled by default (SCALE_AMOUNT). `Time` is left
    unscaled by default (SCALE_TIME) since it is often dropped or
    transformed differently depending on modeling choices in train.py.
    The V1-V28 PCA components are already on a comparable scale and are
    not touched here.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    columns_to_scale = []
    if SCALE_AMOUNT:
        columns_to_scale.append("Amount")
    if SCALE_TIME:
        columns_to_scale.append("Time")

    if not columns_to_scale:
        logger.info("No columns configured for scaling (SCALE_AMOUNT/SCALE_TIME both False).")
        return train_df, val_df, test_df

    scaler = get_scaler()
    logger.info("Fitting %s on columns: %s", SCALER_TYPE, columns_to_scale)

    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    val_df[columns_to_scale] = scaler.transform(val_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])

    save_object(scaler, SCALER_PATH)

    return train_df, val_df, test_df


# ============================================================
# Save
# ============================================================

def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Persist the processed train/val/test splits to data/processed/.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    logger.info("Saved processed train split to %s", TRAIN_DATA_PATH)
    logger.info("Saved processed val split to %s", VAL_DATA_PATH)
    logger.info("Saved processed test split to %s", TEST_DATA_PATH)


# ============================================================
# Orchestration
# ============================================================

def run_preprocessing() -> None:
    """
    Run the full preprocessing pipeline end to end.
    """
    set_random_seed(RANDOM_STATE)

    df = load_raw_data()
    print_dataframe_info(df)

    validate_columns(df)

    df = drop_duplicate_rows(df)

    train_df, val_df, test_df = split_data(df)

    train_df, val_df, test_df = scale_features(train_df, val_df, test_df)

    save_processed_splits(train_df, val_df, test_df)

    logger.info("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_preprocessing()
    except Exception as exc:
        logger.exception("Preprocessing pipeline failed: %s", exc)
        sys.exit(1)