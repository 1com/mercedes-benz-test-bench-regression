from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
OUTLIER_ID = 1770  # y=265.32, ~13 std devs above the mean (see notebooks/01_eda.ipynb)
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
