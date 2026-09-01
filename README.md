# Mercedes-Benz Test Bench Time Prediction

**Value of Product:** Reduce time cars spend on the physical testing bench during manufacturing,
speeding up production without sacrificing safety/quality testing.

**Prediction:** How long (in seconds) a given car configuration will spend on the test bench.

**Evaluation Metric:** R² (coefficient of determination).

**Baseline Model:** Predict the mean `y` for every car → R² ≈ 0.00.

Data: [Kaggle — Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing).
Train ≈ 4,209 rows × 378 columns (`ID`, `y`, 8 categorical `X0`–`X8`, 368 binary flags). Test set
has the same columns minus `y`.

## EDA findings

| # | Finding | So what for modelling |
|---|---|---|
| 1 | No missing values anywhere in train or test. | No imputation strategy needed — one less thing to get wrong. |
| 2 | `y` ranges ~72–160s for the bulk of the data (mean≈100.7, std≈12.7), with one extreme outlier: `ID 1770`, `y=265.32` (~13 std devs above the mean). | Left in, this single point will distort any model trained with squared-error loss (including R² itself). **Decision: drop this row from training** — see Decision Log. |
| 3 | The 8 categorical columns range from 4 (`X4`) to 47 (`X0`) unique values. `X0`, `X2`, `X5` have values in test that never appear in train (6, 6, 4 unseen values respectively). | Any encoder must handle unseen categories gracefully — no plain `LabelEncoder` or one-hot with `handle_unknown="error"`. We use `OneHotEncoder(handle_unknown="ignore")` / ordinal encoding with an explicit unknown value. |
| 4 | Of the 368 binary flag columns, 12 are constant (same value for every row) and 45 are exact duplicates of another column — 57 columns carrying zero unique signal. | Drop before modelling (`src/cleaning.py`) — fewer columns, faster training, no change in information. |
| 5 | **Headline finding:** predicting the average bench time *per `X0` category*, with no model at all, explains ~59% of the variation in `y` (R²≈0.587, 5-fold CV). A full Ridge regression using all 311 cleaned binary columns plus every categorical does *not* beat this (R²≈0.562). | Bench time is dominated by a small number of configuration attributes (`X0` above all), while the hundreds of individual binary flags add very little on their own. This should be the center of the product narrative. |

Figures: `reports/figures/01_y_distribution.png`, `02_y_by_X0_boxplot.png`,
`03_categorical_cardinality.png`, `04_baseline_ladder_results.png`.

## Baseline ladder (M3)

One shared `evaluate()` helper (`src/evaluate.py`) — 5-fold `KFold`, `random_state=42` — scores
every model identically. See `notebooks/03_baseline_ladder.ipynb` for the full run.

| Model | Mean R² (5-fold CV) | Std R² |
|---|---|---|
| 1. Predict the mean | -0.0025 | 0.0027 |
| 2. `X0` group-mean | **0.5870** | 0.0321 |
| 3. Ridge (one-hot categoricals + cleaned binaries) | 0.5621 | 0.0367 |
| 4. XGBoost, default params, cleaned data | 0.5031 | 0.0249 |

For reference, Hashim's original XGBoost run (`notebooks/02_baseline_model.ipynb`, single 80/20
split, uncleaned data) scored R²=0.4493. Re-running the identical model through the shared
`evaluate()` function on the cleaned data (row 4 above) raised that to 0.5031 — cleaning the data
genuinely helped — but it is still the weakest of the four, behind both simple baselines. This
matches the project brief's M5 expectation: a lot of the variance in this dataset looks genuinely
irreducible from these features, and an untuned model may not clear a simple baseline by much.

## Cross-check against Hashim's parallel work

Hashim independently pushed overlapping experiments to `feature/eda-baseline` (dummy baseline,
column cleanup, XGBoost refit) the same day. Comparing results:

| Check | This branch | Hashim's `feature/eda-baseline` | Note |
|---|---|---|---|
| Dummy/mean baseline R² | -0.0025 | -0.0034 | Consistent — both correctly ~0 |
| Columns to drop | 57 (12 constant + 45 duplicate), computed from full `train.csv` | 61 (13 constant + 48 duplicate), computed from his 80% train split | Different computation base — a column can look constant/duplicate in an 80% sample without being so in the full data. Needs a team decision on which to standardize on. |
| Effect of column cleanup alone on XGBoost | — | No change (0.4493 → 0.4493, same single split) | Confirms tree models mostly ignore useless columns on their own — the jump to 0.5031 in this branch's results is mainly from switching to 5-fold CV, not from cleaning |
| Train vs. validation R² gap | Not yet checked here | Train=0.8905, Val=0.4493, gap=0.4412 | **New finding from Hashim** — confirms XGBoost is overfitting badly on a single split, which explains why it underperforms the simple baselines above. Directly relevant to M5 tuning. |

## Repository structure

```
data/raw/            train.csv, test.csv (gitignored — download separately)
notebooks/
  01_eda.ipynb            EDA (Hashim)
  02_baseline_model.ipynb First XGBoost model, exploratory (Hashim)
  03_baseline_ladder.ipynb Required M3 baseline ladder
src/
  cleaning.py          Drop-list logic, outlier handling, random seed
  evaluate.py           Shared 5-fold CV scoring function
  baselines.py          The three baseline models
reports/
  baseline_ladder_results.csv
  figures/
```

## Reproducing

```
uv sync
uv run kaggle competitions download -c mercedes-benz-greener-manufacturing -p data/raw --unzip
uv run jupyter lab
```
