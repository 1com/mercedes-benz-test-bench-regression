# Mercedes-Benz Test Bench Regression

Kaggle "Mercedes-Benz Greener Manufacturing" competition. After a car is assembled, it goes
through a **test bench** — a rig that runs safety/quality checks — before it can leave the
factory. The car occupies the bench for the whole test, so the faster it passes, the more cars
can flow through per day. Mercedes recorded, for thousands of past cars, an anonymized list of
configuration features (`X0, X1, X2, ...` — anonymized so the actual meaning isn't disclosed) and
how many seconds each car spent on the bench (`y`). The task: predict `y` from the features.

This is a **regression** problem (predicting a continuous number, not a category), scored by
**R²** (R-squared) — see [Metrics used throughout](#metrics-used-throughout) below for exactly
what that means and how to read it.

## Setup

1. Install dependencies: `uv sync`
2. Get the dataset (not committed — see `.gitignore`) and place these three files in `data/raw/`:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

   Download them from the [competition page](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/data)
   or via the Kaggle CLI (`kaggle competitions download -c mercedes-benz-greener-manufacturing`).
3. Launch notebooks: `uv run jupyter lab`, and run them in order (01 → 02 → 03). Each is
   self-contained given the data files above.

## Project structure

```
src/
  config.py         constants shared by everything below (RANDOM_STATE, OUTLIER_ID, etc.)
  data.py           raw CSV loading + outlier removal
  preprocessing.py  cleaning, encoding, and get_processed_data() — the shared pipeline
notebooks/
  01_eda.ipynb              exploratory analysis
  02_baseline_model.ipynb   first, deliberately simple model (Linear Regression)
  03_model_comparison.ipynb sweep of regression models on the same shared dataset
```

## The data

`data/raw/train.csv` and `data/raw/test.csv`, 4209 rows each. Every row is one car. `train.csv`
includes the answer (`y`); `test.csv` doesn't — predicting `y` for those rows is the actual task.
Each row has 378 columns in train (377 in test), made up of:

- **`ID`** — a row identifier, no predictive value, dropped before modeling.
- **`y`** — the target (only in `train.csv`). Ranges ~72–170 seconds for nearly every row.
- **8 categorical (text) columns** — `X0, X1, X2, X3, X4, X5, X6, X8`. Each holds a short
  anonymized code (e.g. `"az"`, `"k"`). Cardinality ranges from 4 unique values (`X4`) to 47+ (`X0`).
- **368 binary (0/1) columns** — almost certainly "does this car have option/component X."

### Three problems this dataset has, and how the pipeline handles each

| Problem | Where it's handled | How |
|---|---|---|
| One extreme outlier: row `ID=1770` has `y=265.32`, ~13 std devs above the mean — everything else falls in 72–170 | `src/data.py::load_raw_data()` | Dropped from `df_train` before anything else happens (never touches `df_test` — Kaggle's real test set stays authentic) |
| `X0`, `X2`, `X5` have category values in `test.csv` that **never appear in `train.csv`** — a naive encoder would error or silently mishandle these | `src/preprocessing.py::build_preprocessor()` | `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` — unseen categories get coded `-1` instead of crashing |
| Some binary columns are **constant** (same value every row, no signal) or **exact duplicates** of another column (redundant) | `src/preprocessing.py::find_columns_to_drop()` | Computed from `X_train` only, dropped from train/val/test. In the current 80/20 split, this drops 66 of the 376 feature columns (376 → 310) |

## The pipeline (`src/`)

**The core discipline: calculate on train, apply everywhere.** Any step that "learns" something
from the data — which columns are constant/duplicate, the category→number encoding — learns it
from `X_train` only, then applies that same fixed rule to `X_train`, `X_val`, and `X_test` alike.
Deciding these things from validation or test data would be **data leakage**: it would make the
validation score look better than it honestly should, because the validation set is no longer a
fair, untouched test of generalization — some of its structure quietly shaped the pipeline. Since
the real `test.csv` (or the real world) never gets to leak into training that way, a leaky pipeline
looks great internally and then underperforms once it matters.

| Step | Learned from | Applied to |
|---|---|---|
| Drop the `ID=1770` outlier | nothing to learn — it's a fixed, known row | `df_train` only |
| Train/val split | — | splits `X`/`y` into `X_train`/`X_val` (`X_test` is a separate file, always held apart) |
| Which columns are constant/duplicate | `X_train` only | dropped from `X_train`, `X_val`, **and** `X_test` |
| Categorical → number mapping (`OrdinalEncoder`) | `X_train` only (`.fit(X_train)`) | applied (`.transform()`) to `X_train`, `X_val`, **and** `X_test` |

**`get_processed_data()`** in `src/preprocessing.py` runs this whole sequence in one call and
returns a `ProcessedData` object (`X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `test_ID`) —
the single source of truth for the modeling dataset, so every notebook and every teammate works
from an identical dataframe rather than re-deriving this logic by hand. `get_cv_splitter()`
likewise returns one shared `KFold(n_splits=5, shuffle=True, random_state=42)`, so every model
anyone evaluates is scored against the exact same 5 folds and the numbers stay comparable.

## Metrics used throughout

**R²** is defined as:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

- $SS_{res}$ — the model's actual squared errors (actual `y` − predicted, squared, summed).
- $SS_{tot}$ — the squared errors you'd get by always predicting the **mean of `y`** — the
  simplest possible "model." This is why R²=0 specifically means "no better than guessing the
  mean": if your predictions *are* the mean, $SS_{res}=SS_{tot}$ and the formula gives exactly 0.
  Positive R² means better than that; negative means worse.
- R² is **not** "percent correct" — it's the proportion of `y`'s variance the model explains.
  R²=0.55 means 55% of the variance is explained, not that 55% of predictions are exactly right.
- Sanity-checked in `02_baseline_model.ipynb`'s "Dummy Baseline" cell: `DummyRegressor(strategy='mean')`
  literally always predicts the mean and ignores every feature — it scores R²≈0, confirming the
  metric behaves as expected before trusting any real model's score.

**RMSE** (root mean squared error) is a secondary metric in the same units as `y` (seconds) — "the
model's predictions are off by about 8.7 seconds on average" is a more intuitive read than R² alone.

**`val_r2` vs. `cv_r2_mean`** (both appear throughout notebooks 02/03):
- `val_r2` — R² on one fixed 80/20 train/val split (`X_val`, `random_state=42`). Real and honest,
  but reflects the luck of exactly which rows landed in that one holdout.
- `cv_r2_mean` — the average R² across **5-fold cross-validation**: the training data is split
  into 5 chunks, and 5 rounds are run where each chunk takes a turn as the held-out test set while
  the other 4 are trained on. Averaging over 5 different holdouts smooths out the luck of any one
  split, making it the more reliable number — `03_model_comparison.ipynb`'s results table is
  sorted by `cv_r2_mean` for this reason. A big gap between `val_r2` and `cv_r2_mean` for a given
  model is a hint that its performance is sensitive to exactly which rows it's tested on.

**`train_r2` and `gap`** — the overfitting/underfitting check, run for every model in both
notebooks (`02_baseline_model.ipynb` for its one baseline; `03_model_comparison.ipynb` for all 11):
- `train_r2` — R² measured on the same rows the model trained on. Not very informative alone: a
  flexible enough model can fit its own training data extremely well even when that pattern
  doesn't generalize at all (`DecisionTreeRegressor`'s train_r2=0.975 is a stark example below).
- `gap` — `train_r2 - val_r2`, the actual diagnostic. **Overfitting**: train_r2 high, gap large —
  the model memorized training-specific patterns rather than learning something that generalizes.
  **Underfitting**: gap small, but both train_r2 *and* val_r2 are low — the model isn't even
  capturing the pattern in the data it trained on. **Good fit**: gap small, both scores reasonably
  high.

## Notebooks

**`01_eda.ipynb`** — establishes everything in [The data](#the-data) above, plus a dummy CV
baseline (R²≈0, the sanity floor).

**`02_baseline_model.ipynb`** — the first real model, deliberately simple on purpose: plain
`LinearRegression` (ordinary least squares, no regularization, no tuning) on the shared dataset
from `get_processed_data()`. Result: **Validation R²=0.5131, RMSE=8.74**. Also checks the
train-vs-validation R² gap (train R²=0.6324 vs. val R²=0.5131, gap=0.12) — a large gap would mean
the model is overfitting (memorizing training-specific patterns instead of generalizing); this
gap is modest, as expected from a simple, low-flexibility model.

**`03_model_comparison.ipynb`** — a broader sweep on the identical dataset and CV folds, so every
number below is directly comparable. Regularized linear models (Ridge/Lasso/ElasticNet) use a
`StandardScaler`-scaled copy of the features (`scale_features()`) since they're magnitude-sensitive,
as are the instance/kernel-based models (KNN, SVR); `DecisionTreeRegressor` and the tree ensembles
use the unscaled features since splits don't care about magnitude. Results, sorted by `cv_r2_mean`:

| model | cv_r2_mean | cv_r2_std | val_r2 | train_r2 | gap |
|---|---|---|---|---|---|
| CatBoostRegressor | 0.573 | 0.024 | 0.533 | 0.785 | 0.252 |
| LGBMRegressor | 0.569 | 0.025 | 0.548 | 0.770 | 0.222 |
| Lasso | 0.550 | 0.028 | 0.536 | 0.553 | 0.018 |
| ElasticNet | 0.549 | 0.028 | 0.541 | 0.561 | 0.020 |
| Ridge | 0.547 | 0.022 | 0.513 | 0.632 | 0.119 |
| LinearRegression | 0.544 | 0.022 | 0.513 | 0.632 | 0.119 |
| RandomForestRegressor | 0.538 | 0.033 | 0.495 | 0.918 | 0.423 |
| XGBRegressor | 0.504 | 0.043 | 0.480 | 0.884 | 0.404 |
| SVR | 0.446 | 0.027 | 0.456 | 0.488 | 0.032 |
| KNeighborsRegressor | 0.412 | 0.020 | 0.395 | 0.610 | 0.215 |
| DecisionTreeRegressor | 0.169 | 0.096 | 0.148 | 0.975 | 0.827 |

`gap` is `train_r2 - val_r2` (see [Metrics used throughout](#metrics-used-throughout)) — the same
overfitting diagnostic `02_baseline_model.ipynb` runs for its baseline, applied here to every model.

**Takeaways:** CatBoost/LightGBM win on CV R², but only modestly over the entire linear family —
every linear model here beats *default*-hyperparameter XGBoost and RandomForest. KNN/SVR trail
clearly, likely because "distance between rows" gets unreliable across 310 mostly-binary
dimensions (the curse of dimensionality). `DecisionTreeRegressor` trails everything by a wide
margin (CV R²=0.17) with by far the largest `gap` (0.83) and `cv_r2_std` (0.096) — a single
unconstrained tree splits until it nearly memorizes the training data (train R²=0.975), and that
shows up both as an inflated train score and as unreliable fold-to-fold performance;
`RandomForest`/boosting fix exactly this by averaging many such trees together, though not
entirely — `RandomForest` (gap=0.42) and `XGBoost` (gap=0.40) still overfit substantially, and even
the two CV-winners aren't overfitting-free (`CatBoost`/`LightGBM` gaps of 0.25/0.22). The
regularized linear models have the smallest gaps of the whole sweep (`Lasso`=0.018,
`ElasticNet`=0.020, `SVR`=0.032) — the best-generalizing models here aren't the best-scoring ones,
which is exactly the kind of tradeoff hyperparameter tuning (reining in the tree models'
flexibility) is meant to address. Since everything here uses default hyperparameters, the biggest
lever left is tuning the boosting models (see [Next steps](#next-steps)), not trying more model
types.

## Important caveat: internal CV score vs. the real Kaggle leaderboard

**Our best internal `cv_r2_mean` (0.573) should not be read as "beating" the competition's
public leaderboard** (informally documented elsewhere as topping out around 0.55–0.58) — that
would be comparing two different measurements. Our CV score is computed entirely from slices of
`train.csv`; the real leaderboard score is computed by Kaggle against `test.csv`'s actual `y`
values, which we never see. Two reasons to expect our internal number is optimistic by comparison:

1. This competition is documented to have rows that are *identical* in every feature but have
   *different* `y` — i.e. some of `y`'s variance is pure label noise, unpredictable from features
   at all. That caps how high any model can honestly score, and our train/val split draws both
   sides from the same noisy pool, unlike the real test set.
2. Top leaderboard solutions relied on real feature engineering (`01_eda.ipynb`'s own notes flag
   dimensionality reduction — PCA/ICA/SVD/GRP/SRP on the binary block — as "the differentiator in
   top solutions") plus tuning and ensembling. Matching or beating that with zero tuning would be
   a surprising result, which is itself a reason for skepticism rather than confidence.

The only genuinely comparable number is a real submission: predict on `data.X_test`, pair with
`test_ID`, format per `sample_submission.csv`, and submit to Kaggle.

## Starting a new notebook (hyperparameter tuning, feature engineering, submissions, ...)

Every notebook — this one included when it was written — starts with the same boilerplate,
reusing the shared pipeline instead of re-implementing loading/cleaning/encoding inline:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from src.preprocessing import get_processed_data, get_cv_splitter

data = get_processed_data()
kfold = get_cv_splitter()
```

**Always import from `src` — never copy-paste-and-adapt the preprocessing steps into a new
notebook, even if it looks equivalent.** We found out why the hard way: a teammate's branch
(`eve-feature-engineering`) independently re-implemented this same pipeline by hand — same
outlier, same column-cleanup rule, same encoder, `random_state=42` throughout. It dropped the
outlier *after* the train/val split instead of before. That single ordering difference was enough
to shift which rows `train_test_split` assigned to `X_val` so much that only 42% of the two
"identical `random_state=42`" validation sets actually overlapped — producing a Linear Regression
score of 0.5391 there vs. 0.5131 here, entirely from data-handling drift, not a real modeling
difference. Two pipelines that both claim `random_state=42` reproducibility are only actually
interchangeable if they call the exact same code — that's the whole reason `get_processed_data()`
exists.

**Generating a Kaggle submission** from any fitted model, once you're ready to get a real,
externally-validated score (see the caveat above):
```python
predictions = model.predict(data.X_test)
submission = pd.DataFrame({'ID': data.test_ID, 'y': predictions})
submission.to_csv('../submissions/my_submission.csv', index=False)  # matches sample_submission.csv's columns
```
(`submissions/` is gitignored — create it locally, it isn't committed.)

## Next steps

- **Submit a real prediction** to get an honest, externally-validated score (see caveat above and
  the snippet just above).
- **Hyperparameter tuning** — a natural next notebook (`04_hyperparameter_tuning.ipynb`): reuse
  `get_processed_data()` and `get_cv_splitter()` unchanged so tuned results stay comparable to the
  table above; pick one model to start with (LightGBM/CatBoost to push the leaders further, or
  XGBoost since it underperformed its potential with defaults); use `GridSearchCV` or
  `RandomizedSearchCV` with `cv=kfold` (the same shared splitter); compare `.best_score_` against
  that model's `cv_r2_mean` row above; do one final, untouched check against `X_val`/`y_val` after
  picking final hyperparameters, since repeatedly searching against the same 5 CV folds risks
  quietly overfitting the hyperparameters to those folds specifically.
- **Feature engineering** on the binary block (dimensionality reduction — PCA/ICA/SVD/GRP/SRP, per
  the EDA notes above). To keep every notebook working from the same final dataset, this belongs
  *inside* the shared pipeline, not notebook-local code: either extend
  `src/preprocessing.py::get_processed_data()` or add a new `src/features.py` step it calls. Follow
  the same calculate-on-train/apply-everywhere discipline as everything else — fit any reducer
  (e.g. `PCA`) on `X_train` only, then `.transform()` `X_train`/`X_val`/`X_test`.
