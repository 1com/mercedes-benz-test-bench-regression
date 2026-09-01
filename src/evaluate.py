"""The ONE shared scoring function for this project.

Every model number that reaches the slides must come from this function, so
that scores are directly comparable across the baseline ladder and the final
model. Uses 5-fold KFold cross-validation with a fixed random seed (see
src/cleaning.py:RANDOM_SEED) instead of a single train/validation split, so a
model's score isn't a fluke of one lucky/unlucky split.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from src.cleaning import RANDOM_SEED


def evaluate(estimator, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Score `estimator` with n_splits-fold cross-validation.

    `estimator` must be an unfitted sklearn-compatible model (implements
    fit/predict). A fresh clone is fit on each fold so no fold leaks into
    another's training data.

    Returns a dict with per-fold scores plus the mean/std R2 that should be
    quoted anywhere this model's performance is reported.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = clone(estimator)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        fold_scores.append(r2_score(y_val, preds))

    fold_scores = np.array(fold_scores)
    return {
        "fold_scores": fold_scores.tolist(),
        "mean_r2": fold_scores.mean(),
        "std_r2": fold_scores.std(),
    }
