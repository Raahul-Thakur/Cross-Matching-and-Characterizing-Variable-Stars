"""Shared utilities for paths, logging, and reproducibility."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import DATA_DIR, INTERIM_DIR, LIGHTCURVE_CACHE_DIR, MODEL_DIR, OUTPUT_DIR, PLOT_DIR, RAW_DIR


def ensure_directories() -> None:
    """Create project data and output directories if they do not exist."""
    for path in (DATA_DIR, RAW_DIR, INTERIM_DIR, LIGHTCURVE_CACHE_DIR, OUTPUT_DIR, PLOT_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_source_id(series):
    """Keep Gaia source IDs stable when moving between pandas/csv/json."""
    return series.astype("Int64").astype(str)


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan
