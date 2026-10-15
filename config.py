"""
DAMO510 Take-Home Assignment 2
Central configuration: seed, paths, column names, and small shared helpers.

Everything random in this project traces back to SEED defined here.
Nothing downstream is allowed to hard-code a different value.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 1. THE ASSIGNED SEED
# --------------------------------------------------------------------------
SEED = 93  # assigned analysis seed; used for the split, CV shuffling,
           # every model random_state, SMOTE, and the robustness resampling.

# --------------------------------------------------------------------------
# 2. PATHS
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
SRC = ROOT / "src"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"
TABLES = OUT / "tables"
LOGS = OUT / "logs"
STORE = OUT / "store"          # small JSON results passed between stages
REPORT = ROOT / "report"
NOTEBOOKS = ROOT / "notebooks"

for _p in (DATA_RAW, DATA_PROC, OUT, FIGS, TABLES, LOGS, STORE, REPORT, NOTEBOOKS):
    _p.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW / "ai4i2020_raw.csv"
MODEL_READY_CSV = DATA_PROC / "ai4i2020_model_ready.csv"

# --------------------------------------------------------------------------
# 3. DATA SOURCE
# --------------------------------------------------------------------------
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
UCI_DATASET_ID = 601  # ucimlrepo fallback

EXPECTED_SHAPE = (10000, 14)

# --------------------------------------------------------------------------
# 4. COLUMN NAMES (as they appear in the untouched UCI file)
# --------------------------------------------------------------------------
TARGET = "Machine failure"
ID_COLS = ["UDI", "Product ID"]
FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
DETERMINISTIC_FLAGS = ["TWF", "HDF", "PWF", "OSF"]   # the four modes AI4I ORs together
CATEGORICAL = ["Type"]
NUMERIC_RAW = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Short machine-friendly aliases used after cleaning
RENAME = {
    "Air temperature [K]": "air_temp_K",
    "Process temperature [K]": "process_temp_K",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_Nm",
    "Tool wear [min]": "tool_wear_min",
    "Type": "type",
    "Machine failure": "machine_failure",
}
NUMERIC_CLEAN = [
    "air_temp_K",
    "process_temp_K",
    "rot_speed_rpm",
    "torque_Nm",
    "tool_wear_min",
]
CAT_CLEAN = ["type"]
TARGET_CLEAN = "machine_failure"

ENGINEERED = ["power_W", "temp_diff_K", "wear_torque"]

# --------------------------------------------------------------------------
# 5. MODELLING CONSTANTS
# --------------------------------------------------------------------------
TEST_SIZE = 0.20
CV_FOLDS = 5
SELECTION_METRIC = "average_precision"   # PR-AUC; declared before any search runs
SELECTION_METRIC_LABEL = "PR-AUC (average precision)"

# Cost framing used in the threshold analysis (stated as an assumption).
COST_FN = 500.0   # cost of one missed failure (unplanned stoppage)
COST_FP = 50.0    # cost of one false alarm (needless inspection)
COST_RATIO = COST_FN / COST_FP

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
ROBUSTNESS_SEEDS = [93, 7, 21, 42, 55, 101, 202, 314, 777, 1234]

FIG_DPI = 300

# --------------------------------------------------------------------------
# 6. SHARED HELPERS
# --------------------------------------------------------------------------


class Tee:
    """Write console output to both stdout and a per-stage log file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = open(self.path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def start_log(stage: str) -> Tee:
    """Redirect stdout to outputs/logs/<stage>.log as well as the console."""
    tee = Tee(LOGS / f"{stage}.log")
    sys.stdout = tee
    return tee


def end_log(tee: Tee) -> None:
    sys.stdout = tee.stdout
    tee.close()


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def save_table(df, name: str, index: bool = False) -> Path:
    """Persist a table to outputs/tables/ as CSV and echo it to the log."""
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"\n[table] {path.name}")
    print(df.to_string(index=index))
    return path


def save_fig(fig, name: str) -> Path:
    """Persist a figure to outputs/figures/ at the configured dpi."""
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"[figure] {path.name}")
    return path


def put(key: str, value) -> None:
    """Store a scalar/dict result for later stages and for the report builder."""
    path = STORE / f"{key}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, default=str)


def get(key: str):
    path = STORE / f"{key}.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def setup_matplotlib():
    """Non-interactive backend and a consistent house style for every figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIG_DPI,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": True,
        "figure.autolayout": False,
    })
    return plt, sns


# Colour-blind-safe pair used consistently: no failure / failure
C_OK = "#4C78A8"
C_FAIL = "#E45756"
PALETTE = ["#4C78A8", "#E45756", "#72B7B2", "#F58518", "#54A24B", "#B279A2"]
