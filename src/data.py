from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, OUTLIER_ID


def load_raw_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train/test CSVs and drop the known outlier row (ID=OUTLIER_ID) from train only."""
    df_train = pd.read_csv(data_dir / "train.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    df_train = df_train[df_train["ID"] != OUTLIER_ID].reset_index(drop=True)

    return df_train, df_test


def split_features_target(df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df_train["y"]
    X = df_train.drop(["ID", "y"], axis=1)
    return X, y


def get_test_features(df_test: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    test_ID = df_test["ID"]
    X_test = df_test.drop(["ID"], axis=1)
    return test_ID, X_test
