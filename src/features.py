"""Optional feature engineering step: PCA on the binary (originally 0/1) feature block.

Flagged since 01_eda.ipynb as a candidate ("differentiator in top solutions"), never applied.
Only compresses the binary columns, not the categorical ones (cat__X0 etc, already
ordinal-encoded integers) -- PCA assumes continuous, correlated inputs, which a column of
arbitrary category codes doesn't satisfy. Follows the same calculate-on-train/apply-everywhere
discipline as src/preprocessing.py: the PCA is fit on X_train's binary columns only, then applied
(never re-fit) to X_val and X_test.
"""

import pandas as pd
from sklearn.decomposition import PCA

from src.config import RANDOM_STATE
from src.preprocessing import ProcessedData


def reduce_binary_features(
    data: ProcessedData, variance_to_keep: float = 0.95, random_state: int = RANDOM_STATE
) -> ProcessedData:
    """Replace the binary feature block with enough PCA components to retain
    `variance_to_keep` of its variance (fit on X_train only). Categorical columns pass through
    unchanged."""
    cat_cols = [c for c in data.X_train.columns if c.startswith("cat__")]
    binary_cols = [c for c in data.X_train.columns if c.startswith("num__")]

    pca = PCA(n_components=variance_to_keep, random_state=random_state)
    pca.fit(data.X_train[binary_cols])

    def transform(X: pd.DataFrame) -> pd.DataFrame:
        components = pca.transform(X[binary_cols])
        component_cols = [f"pca_{i}" for i in range(components.shape[1])]
        components_df = pd.DataFrame(components, columns=component_cols, index=X.index)
        return pd.concat([X[cat_cols], components_df], axis=1)

    return ProcessedData(
        X_train=transform(data.X_train),
        X_val=transform(data.X_val),
        X_test=transform(data.X_test),
        y_train=data.y_train,
        y_val=data.y_val,
        test_ID=data.test_ID,
    )
