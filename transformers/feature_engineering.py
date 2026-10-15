"""
Transformer block: physics-based feature engineering.

Each feature is a row-wise function of that row's own sensor values, so it uses
no cross-row statistic and cannot leak information across a train/test split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with three engineered features appended.

    - ``power_W``: mechanical power, torque * angular velocity (rpm -> rad/s).
    - ``temp_diff_K``: process minus air temperature (heat not carried away).
    - ``wear_torque``: tool wear * torque, a proxy for accumulated overstrain.
    """
    out = df.copy()
    out["power_W"] = out["torque_Nm"] * out["rot_speed_rpm"] * 2 * np.pi / 60.0
    out["temp_diff_K"] = out["process_temp_K"] - out["air_temp_K"]
    out["wear_torque"] = out["tool_wear_min"] * out["torque_Nm"]
    logger.info("Added engineered features: power_W, temp_diff_K, wear_torque")
    return out
