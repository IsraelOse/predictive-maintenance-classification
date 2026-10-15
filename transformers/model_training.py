"""
Transformer block: train and evaluate the four classifiers.

Builds each model as preprocessing + estimator in one sklearn Pipeline (so the
preprocessor is fitted only on training folds), fits on the training partition,
and scores once on the held-out test set. Returns fitted pipelines and a tidy
metrics table. Hyperparameters come from config (selected offline; see README).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils.config import MODEL_PARAMS, MODEL_ORDER, SEED
from utils.logging_config import get_logger
from transformers.preprocessing import build_preprocessor

logger = get_logger(__name__)


def _build_estimator(name: str, params: Dict[str, Any], seed: int):
    """Instantiate one estimator by name with its configured parameters."""
    if name == "Logistic Regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if name == "XGBoost":
        return XGBClassifier(random_state=seed, eval_metric="logloss",
                             n_jobs=-1, tree_method="hist", **params)
    raise ValueError(f"Unknown model: {name}")


def _score(name: str, y_true, y_pred, y_proba) -> Dict[str, Any]:
    """One row of the shared evaluation framework at the 0.5 cut-off."""
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
    }


def train_and_evaluate(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    seed: int = SEED,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """Fit every configured model and return (fitted_pipelines, metrics_table)."""
    fitted: Dict[str, Pipeline] = {}
    rows = []
    for name in MODEL_ORDER:
        scale = name == "Logistic Regression"
        pipe = Pipeline([
            ("pre", build_preprocessor(scale=scale)),
            ("clf", _build_estimator(name, MODEL_PARAMS[name], seed)),
        ])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = _score(name, y_test, pred, proba)
        fitted[name] = pipe
        rows.append(row)
        logger.info("%-20s ROC-AUC %.4f  PR-AUC %.4f  recall %.4f",
                    name, row["roc_auc"], row["pr_auc"], row["recall"])
    metrics = pd.DataFrame(rows)
    return fitted, metrics
