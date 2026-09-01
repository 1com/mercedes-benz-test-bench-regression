"""The three required baseline models, in increasing order of sophistication.
Each is a plain sklearn-compatible estimator (fit/predict), so all three can
be scored identically through src.evaluate.evaluate().

  1. DummyRegressor(strategy="mean")  -- predict the overall average for every row
  2. X0GroupMeanRegressor              -- predict the average y for that row's X0 value
  3. build_ridge_baseline()            -- Ridge regression on one-hot categoricals
                                          + the cleaned binary columns
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.cleaning import RANDOM_SEED


class X0GroupMeanRegressor(BaseEstimator, RegressorMixin):
    """Predicts the mean y for a row's X0 value (its training-time average).
    Falls back to the overall training mean for any X0 value never seen
    during fit (mirrors the real-world case: test.csv contains X0 values
    that never appear in train.csv)."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = y.mean()
        self.group_means_ = y.groupby(X["X0"]).mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["X0"].map(self.group_means_).fillna(self.global_mean_).to_numpy()


def build_ridge_baseline(cat_cols: list[str], num_cols: list[str], alpha: float = 1.0) -> Pipeline:
    """Ridge regression on one-hot encoded categoricals + passthrough binary columns.
    handle_unknown='ignore' means an X0/X2/X5 value never seen in training
    (a real feature of this dataset) doesn't crash the encoder -- it's encoded
    as all-zeros instead."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("ridge", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
    ])
