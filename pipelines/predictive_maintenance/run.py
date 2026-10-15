"""
Pipeline orchestration: predictive_maintenance.

Wires the decoupled blocks in order:
    data_loaders -> transformers (clean -> engineer -> split -> train/eval)
    -> data_exporters
Each step passes its output to the next; no block reaches into another's
internals. Run from the repository root:  python -m pipelines.predictive_maintenance.run
"""
from __future__ import annotations

import pandas as pd

from data_loaders.ai4i_loader import load_raw_data
from transformers.clean import clean
from transformers.feature_engineering import add_engineered_features
from transformers.split import split_data
from transformers.model_training import train_and_evaluate
from data_exporters.local_exporter import export_model_ready, export_metrics
from utils.config import SEED
from utils.logging_config import get_logger

logger = get_logger("pipelines.predictive_maintenance")


def run() -> pd.DataFrame:
    """Execute the full pipeline and return the model comparison table."""
    logger.info("=== predictive_maintenance pipeline start (seed=%d) ===", SEED)

    # 1. Load
    raw = load_raw_data()

    # 2. Transform: clean -> engineer features
    model_ready = add_engineered_features(clean(raw))
    export_model_ready(model_ready)

    # 3. Transform: split -> train & evaluate
    X_train, X_test, y_train, y_test = split_data(model_ready)
    _, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 4. Export
    export_metrics(metrics)

    best = metrics.loc[metrics["pr_auc"].idxmax()]
    logger.info("Best model by PR-AUC: %s (PR-AUC %.4f, ROC-AUC %.4f)",
                best["model"], best["pr_auc"], best["roc_auc"])
    logger.info("=== pipeline complete ===")
    return metrics


if __name__ == "__main__":
    print(run().to_string(index=False))
