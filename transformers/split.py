"""
Transformer block: the stratified train/test split.

Pure function of (DataFrame, seed): returns four frames/arrays. Stratifying on
the target matters at a ~3.4% base rate so the test failure count is stable.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import TARGET_CLEAN, TEST_SIZE, SEED
from utils.logging_config import get_logger

logger = get_logger(__name__)


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE,
               seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test, stratified on the target."""
    y = df[TARGET_CLEAN]
    X = df.drop(columns=[TARGET_CLEAN])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    logger.info("Split: train=%d (pos=%d) test=%d (pos=%d)",
                len(X_train), int(y_train.sum()), len(X_test), int(y_test.sum()))
    return X_train, X_test, y_train, y_test
