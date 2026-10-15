"""
Transformer block: the shared preprocessing factory.

One factory builds the ColumnTransformer used by every model, so the baseline,
the search, and the evaluation provably fit the same transformations. Scaling
is the only model-specific step (needed by logistic regression, inert for
trees). Transformers are returned unfitted so they are fitted inside CV folds.
"""
from __future__ import annotations

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import NUMERIC_CLEAN, ENGINEERED, CAT_CLEAN


def feature_columns(use_engineered: bool = True) -> Tuple[List[str], List[str]]:
    """Return (numeric, categorical) predictor names for a feature set."""
    numeric = list(NUMERIC_CLEAN) + (list(ENGINEERED) if use_engineered else [])
    return numeric, list(CAT_CLEAN)


def build_preprocessor(scale: bool, use_engineered: bool = True) -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer.

    Common to all models: median imputation (a live-sensor-dropout safeguard)
    and one-hot encoding of the three-level product grade. ``scale`` adds
    standardisation, applied only for logistic regression.
    """
    numeric, categorical = feature_columns(use_engineered)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
