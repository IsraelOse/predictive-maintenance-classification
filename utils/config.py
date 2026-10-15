"""
Central configuration for the Predictive Maintenance pipeline.

Everything random in this project traces back to SEED defined here. Paths,
column names, and modelling constants live here so no block hard-codes them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# --- seed -------------------------------------------------------------------
SEED: int = 93

# --- paths ------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROC: Path = ROOT / "data" / "processed"
OUTPUTS: Path = ROOT / "outputs"
TABLES: Path = OUTPUTS / "tables"

RAW_CSV: Path = DATA_RAW / "ai4i2020_raw.csv"
MODEL_READY_CSV: Path = DATA_PROC / "ai4i2020_model_ready.csv"

# --- data source ------------------------------------------------------------
UCI_ZIP_URL: str = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
UCI_DATASET_ID: int = 601
EXPECTED_SHAPE = (10000, 14)

# --- columns (as they appear in the untouched UCI file) ---------------------
TARGET: str = "Machine failure"
ID_COLS: List[str] = ["UDI", "Product ID"]
# Post-hoc failure-mode labels. Four of them OR together to the target, so
# keeping any would leak the answer. Dropped in cleaning.
FAILURE_FLAGS: List[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]

RENAME: Dict[str, str] = {
    "Air temperature [K]": "air_temp_K",
    "Process temperature [K]": "process_temp_K",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_Nm",
    "Tool wear [min]": "tool_wear_min",
    "Type": "type",
    "Machine failure": "machine_failure",
}

NUMERIC_CLEAN: List[str] = [
    "air_temp_K", "process_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min",
]
ENGINEERED: List[str] = ["power_W", "temp_diff_K", "wear_torque"]
CAT_CLEAN: List[str] = ["type"]
TARGET_CLEAN: str = "machine_failure"

# --- modelling constants ----------------------------------------------------
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
SELECTION_METRIC: str = "average_precision"  # PR-AUC

# Final model hyperparameters. These were selected by the randomised search
# documented in the analysis notebook; production trains the chosen config
# rather than re-deriving it every run.
#   The imbalance study retained "no treatment" for the final models, so no
#   class weighting is applied here; that keeps this pipeline consistent with
#   the analysis report and notebook.
MODEL_PARAMS: Dict[str, Dict] = {
    "Logistic Regression": {"max_iter": 5000},
    "Decision Tree": {"max_depth": 6, "min_samples_leaf": 10},
    "Random Forest": {"n_estimators": 400, "min_samples_leaf": 2},
    "XGBoost": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1},
}
MODEL_ORDER: List[str] = [
    "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost",
]
