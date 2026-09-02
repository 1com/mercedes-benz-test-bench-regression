from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import DATA_DIR, N_SPLITS, RANDOM_STATE, TEST_SIZE
from src.data import get_test_features, load_raw_data, split_features_target


@dataclass
class ProcessedData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    test_ID: pd.Series


def find_columns_to_drop(X_train: pd.DataFrame) -> list[str]:
    """Constant and exact-duplicate columns, computed from X_train only."""
    constant_cols = [c for c in X_train.columns if X_train[c].nunique() == 1]

    non_constant_cols = [c for c in X_train.columns if c not in constant_cols]
    dup_groups = X_train[non_constant_cols].T.groupby(
        X_train[non_constant_cols].T.apply(tuple, axis=1)
    ).groups
    duplicate_groups = [list(v) for v in dup_groups.values() if len(v) > 1]
    duplicate_cols_to_drop = [c for group in duplicate_groups for c in group[1:]]

    return constant_cols + duplicate_cols_to_drop


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """OrdinalEncoder for categoricals (safe for unseen test categories), passthrough for numeric."""
    cat_cols = X_train.select_dtypes(include="str").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="str").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def scale_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a StandardScaler on X_train only; for scale-sensitive models (linear/KNN/SVR), not trees."""
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    scaler.fit(X_train)

    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


def get_processed_data(
    data_dir: Path = DATA_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> ProcessedData:
    """Single entry point: load, split, drop constant/duplicate columns, encode. One shared
    final dataset for every model/notebook/teammate."""
    df_train, df_test = load_raw_data(data_dir)
    X, y = split_features_target(df_train)
    test_ID, X_test = get_test_features(df_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    cols_to_drop = find_columns_to_drop(X_train)
    X_train = X_train.drop(columns=cols_to_drop)
    X_val = X_val.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_train_encoded = preprocessor.transform(X_train)
    X_val_encoded = preprocessor.transform(X_val)
    X_test_encoded = preprocessor.transform(X_test)

    return ProcessedData(
        X_train=X_train_encoded,
        X_val=X_val_encoded,
        X_test=X_test_encoded,
        y_train=y_train,
        y_val=y_val,
        test_ID=test_ID,
    )


def get_cv_splitter(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
