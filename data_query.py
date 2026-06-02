"""Gaia TAP queries and Gaia-side feature preparation."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    GAIA_DR,
    GAIA_FALLBACK_SAMPLE_SIZES,
    GAIA_MAX_RUWE,
    GAIA_MIN_PARALLAX_OVER_ERROR,
    GAIA_QUERY_RETRIES,
    GAIA_SAMPLE_SIZE,
    INTERIM_DIR,
    RAW_DIR,
)
from utils import safe_source_id, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)


def gaia_quality_where(
    min_parallax_over_error: float = GAIA_MIN_PARALLAX_OVER_ERROR,
    max_ruwe: float = GAIA_MAX_RUWE,
) -> str:
    return f"""
        gs.parallax > 0
        AND gs.parallax_error IS NOT NULL
        AND gs.parallax / gs.parallax_error >= {min_parallax_over_error}
        AND gs.bp_rp IS NOT NULL
        AND gs.phot_g_mean_mag IS NOT NULL
        AND gs.ruwe IS NOT NULL
        AND gs.ruwe < {max_ruwe}
    """


def build_gaia_query(sample_size: int = GAIA_SAMPLE_SIZE) -> str:
    """Build a Gaia DR3 variable-source query with quality filters."""
    return f"""
    SELECT TOP {sample_size}
        gs.source_id,
        gs.ra,
        gs.dec,
        gs.l,
        gs.b,
        gs.parallax,
        gs.parallax_error,
        gs.phot_g_mean_mag,
        gs.bp_rp,
        gs.ruwe,
        gs.ag_gspphot
    FROM {GAIA_DR}.vari_summary AS vs
    JOIN {GAIA_DR}.gaia_source AS gs
        ON vs.source_id = gs.source_id
    WHERE {gaia_quality_where()}
    """


def _launch_gaia_query_with_retries(query: str, sample_size: int):
    from astroquery.gaia import Gaia

    last_error: Exception | None = None
    for attempt in range(1, GAIA_QUERY_RETRIES + 1):
        try:
            LOGGER.info("Gaia query attempt %s/%s for TOP %s", attempt, GAIA_QUERY_RETRIES, sample_size)
            job = Gaia.launch_job_async(query)
            return job.get_results().to_pandas()
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Gaia query failed on attempt %s/%s: %s", attempt, GAIA_QUERY_RETRIES, exc)
            if attempt < GAIA_QUERY_RETRIES:
                time.sleep(3 * attempt)
    raise RuntimeError(
        "Gaia archive query failed after retries. The archive may be unstable or the query may be too large. "
        "Try a smaller run such as: python CMCVA.py --sample-size 1000 --max-lightcurves 200"
    ) from last_error


def query_gaia_variables(
    sample_size: int = GAIA_SAMPLE_SIZE,
    output_path: Path = RAW_DIR / "gaia_dr3_variables.csv",
    metadata_path: Path = RAW_DIR / "gaia_query_metadata.json",
    force: bool = False,
) -> pd.DataFrame:
    """Query Gaia DR3 variables and cache the raw table."""
    if output_path.exists() and not force:
        LOGGER.info("Loading cached Gaia table from %s", output_path)
        return load_gaia_table(output_path)

    from astroquery.gaia import Gaia

    query = build_gaia_query(sample_size)
    LOGGER.info("Submitting Gaia query for up to %s sources", sample_size)
    try_sizes = [sample_size] + [size for size in GAIA_FALLBACK_SAMPLE_SIZES if size < sample_size]
    last_error: Exception | None = None
    actual_sample_size = sample_size
    for current_size in try_sizes:
        try:
            query = build_gaia_query(current_size)
            gaia_df = _launch_gaia_query_with_retries(query, current_size)
            actual_sample_size = current_size
            break
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Gaia TOP %s query failed; trying smaller fallback if available.", current_size)
    else:
        raise RuntimeError(
            "Could not download Gaia data. Try again later, log into Gaia, or manually provide "
            "data/raw/gaia_dr3_variables.csv."
        ) from last_error
    gaia_df = prepare_gaia_table(gaia_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gaia_df.to_csv(output_path, index=False)
    write_json(
        metadata_path,
        {
            "gaia_data_release": GAIA_DR,
            "query_utc": utc_now_iso(),
            "requested_sample_size": sample_size,
            "actual_query_top": actual_sample_size,
            "quality_filters": {
                "parallax_positive": True,
                "min_parallax_over_error": GAIA_MIN_PARALLAX_OVER_ERROR,
                "max_ruwe": GAIA_MAX_RUWE,
                "valid_bp_rp": True,
            },
            "query": query,
        },
    )
    return gaia_df


def load_gaia_table(path: Path = RAW_DIR / "gaia_dr3_variables.csv") -> pd.DataFrame:
    return prepare_gaia_table(pd.read_csv(path))


def prepare_gaia_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply local quality filters and derive corrected absolute magnitude."""
    required = {
        "source_id",
        "ra",
        "dec",
        "parallax",
        "parallax_error",
        "phot_g_mean_mag",
        "bp_rp",
        "ruwe",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Gaia table is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["source_id"] = safe_source_id(out["source_id"])
    numeric_cols = [
        "ra",
        "dec",
        "l",
        "b",
        "parallax",
        "parallax_error",
        "phot_g_mean_mag",
        "bp_rp",
        "ruwe",
        "ag_gspphot",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    mask = (
        (out["parallax"] > 0)
        & out["parallax_error"].notna()
        & ((out["parallax"] / out["parallax_error"]) >= GAIA_MIN_PARALLAX_OVER_ERROR)
        & out["bp_rp"].notna()
        & out["phot_g_mean_mag"].notna()
        & out["ruwe"].notna()
        & (out["ruwe"] < GAIA_MAX_RUWE)
    )
    out = out.loc[mask].drop_duplicates("source_id").reset_index(drop=True)

    extinction = out["ag_gspphot"].fillna(0.0) if "ag_gspphot" in out.columns else 0.0
    out["abs_mag_g"] = out["phot_g_mean_mag"] + 5.0 * np.log10(out["parallax"]) - 10.0 - extinction
    out["parallax_over_error"] = out["parallax"] / out["parallax_error"]
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(INTERIM_DIR / "gaia_quality_filtered.csv", index=False)
    return out
