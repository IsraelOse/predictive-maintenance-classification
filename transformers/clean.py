"""
Transformer block: clean the raw AI4I data.

Pure function: drops identifiers and the leaking failure-mode flags, renames
sensor columns to machine-friendly names. Accepts a DataFrame, returns a new
DataFrame; no I/O, no global state.
"""
from __future__ import annotations

import pandas as pd

from utils.config import ID_COLS, FAILURE_FLAGS, RENAME
from utils.logging_config import get_logger

logger = get_logger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: identifiers and leakage flags removed, renamed.

    The four deterministic failure flags (TWF, HDF, PWF, OSF) OR together to
    reproduce the target on ~99.9% of rows, so every flag is dropped to prevent
    target leakage.
    """
    drop_cols = ID_COLS + FAILURE_FLAGS
    out = df.drop(columns=drop_cols).rename(columns=RENAME)
    logger.info("Dropped %d columns (ids + leakage flags); %d columns retained",
                len(drop_cols), out.shape[1])
    return out
