"""Reusable data-cleaning logic for the Mercedes-Benz test bench dataset.

Turns the drop-list findings from notebooks/01_eda.ipynb into code that can be
imported and reused, instead of being recomputed by hand in a notebook cell.

Cleaning rules (all thresholds/decisions computed from TRAIN only, then applied
to both train and test, so nothing about the test set leaks into what gets dropped):
  1. Constant columns: binary flag columns where every training row has the same
     value (zero variance -> no signal for any model).
  2. Duplicate columns: binary flag columns that are an exact copy of another
     column. Only the first column in each duplicate group is kept.

RANDOM_SEED is defined once here so every downstream script/notebook can import
it and stay consistent, per the project's "one fixed seed, defined once" rule.
"""

import pandas as pd

RANDOM_SEED = 42

ID_COL = "ID"
TARGET_COL = "y"
OUTLIER_ID = 1770  # y=265.32, ~13 std devs above the mean. Confirmed against
                    # real data on 2026-09-01; Notebook 01 had mislabeled
                    # this as "ID 883" (that was the pandas row position, not the
                    # actual ID column value).


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """The 8 text/category columns. pandas 3.0 loads these as dtype 'str', not
    'object' -- this is the gotcha Hashim flagged in notebook 01."""
    return df.select_dtypes(include=["str"]).columns.tolist()


def get_binary_columns(df: pd.DataFrame) -> list[str]:
    """The ~368 yes/no flag columns: everything except ID, y, and the categoricals."""
    cat_cols = get_categorical_columns(df)
    return [c for c in df.columns if c not in cat_cols and c not in (ID_COL, TARGET_COL)]


def get_constant_columns(train_df: pd.DataFrame, binary_cols: list[str]) -> list[str]:
    """Binary columns with only one unique value in training data."""
    return [c for c in binary_cols if train_df[c].nunique() == 1]


def get_duplicate_columns(train_df: pd.DataFrame, binary_cols: list[str]) -> list[str]:
    """Binary columns that are an exact copy of an earlier column.
    Returns only the columns TO DROP (keeps the first column of each group)."""
    grouped = train_df[binary_cols].T.groupby(
        train_df[binary_cols].T.apply(tuple, axis=1)
    ).groups
    duplicate_sets = [list(v) for v in grouped.values() if len(v) > 1]
    return [c for group in duplicate_sets for c in group[1:]]


def get_drop_list(train_df: pd.DataFrame) -> list[str]:
    """Full list of binary columns to drop: constant + duplicate.
    Must be computed from train_df only, then applied to both train and test."""
    binary_cols = get_binary_columns(train_df)
    constant_cols = get_constant_columns(train_df, binary_cols)
    non_constant = [c for c in binary_cols if c not in constant_cols]
    duplicate_cols = get_duplicate_columns(train_df, non_constant)
    return constant_cols + duplicate_cols


def drop_unwanted_columns(df: pd.DataFrame, drop_list: list[str]) -> pd.DataFrame:
    """Apply a precomputed drop list to any dataframe (train or test)."""
    return df.drop(columns=[c for c in drop_list if c in df.columns])


def drop_outlier_row(train_df: pd.DataFrame, outlier_id: int = OUTLIER_ID) -> pd.DataFrame:
    """Remove the single extreme-outlier row from TRAINING data only.
    Never apply this to the test set -- test rows must never be dropped,
    since predictions are required for every test ID."""
    return train_df[train_df[ID_COL] != outlier_id].reset_index(drop=True)
