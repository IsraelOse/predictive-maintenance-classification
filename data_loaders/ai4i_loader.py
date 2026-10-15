"""
Data loader block: fetch the AI4I 2020 Predictive Maintenance dataset.

Decoupled: this block only acquires raw data and returns a DataFrame. It knows
nothing about cleaning, modelling, or exporting.
"""
from __future__ import annotations

import io
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.config import RAW_CSV, UCI_ZIP_URL, UCI_DATASET_ID, EXPECTED_SHAPE
from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_raw_data(cache_path: Path = RAW_CSV,
                  url: str = UCI_ZIP_URL,
                  dataset_id: int = UCI_DATASET_ID) -> pd.DataFrame:
    """Return the untouched AI4I 2020 dataset as a DataFrame.

    Reads a cached copy if present; otherwise downloads the zip from the UCI
    archive (falling back to the ``ucimlrepo`` package) and caches it. The
    cached copy makes every later run offline and byte-identical.
    """
    if cache_path.exists():
        logger.info("Reading cached raw data: %s", cache_path)
        return pd.read_csv(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading raw data from %s", url)
        blob = urllib.request.urlopen(url, timeout=120).read()
        archive = zipfile.ZipFile(io.BytesIO(blob))
        member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(io.BytesIO(archive.read(member)))
    except Exception as exc:  # pragma: no cover - network fallback
        logger.warning("Direct download failed (%s); using ucimlrepo.", exc)
        from ucimlrepo import fetch_ucirepo
        df = fetch_ucirepo(id=dataset_id).data.original.copy().rename(columns={
            "UID": "UDI",
            "Air temperature": "Air temperature [K]",
            "Process temperature": "Process temperature [K]",
            "Rotational speed": "Rotational speed [rpm]",
            "Torque": "Torque [Nm]",
            "Tool wear": "Tool wear [min]",
        })

    df.to_csv(cache_path, index=False)
    logger.info("Cached raw data to %s (shape %s)", cache_path, df.shape)
    if df.shape != EXPECTED_SHAPE:
        logger.warning("Unexpected shape %s (expected %s)", df.shape, EXPECTED_SHAPE)
    return df
