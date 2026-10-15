"""
Data exporter block: persist pipeline outputs.

Decoupled sink: writes the model-ready dataset and the metrics table to disk.
Swapping this block for a warehouse or cloud-bucket writer would not touch any
loader or transformer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.config import MODEL_READY_CSV, TABLES, DATA_PROC
from utils.logging_config import get_logger

logger = get_logger(__name__)


def export_model_ready(df: pd.DataFrame, path: Path = MODEL_READY_CSV) -> Path:
    """Write the cleaned, feature-engineered dataset to CSV."""
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote model-ready dataset: %s (%d rows)", path, len(df))
    return path


def export_metrics(metrics: pd.DataFrame, name: str = "model_comparison") -> Path:
    """Write the metrics table to CSV and JSON under outputs/tables/."""
    TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES / f"{name}.csv"
    json_path = TABLES / f"{name}.json"
    metrics.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2))
    logger.info("Wrote metrics: %s and %s", csv_path.name, json_path.name)
    return csv_path
