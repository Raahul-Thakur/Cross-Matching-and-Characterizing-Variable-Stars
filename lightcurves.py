"""ASAS-SN light-curve downloading, caching, and cleaning."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (
    LIGHTCURVE_CACHE_DIR,
    MIN_LIGHTCURVE_POINTS,
    REQUEST_BACKOFF_SECONDS,
    REQUEST_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    SKYPATROL_RADIUS_ARCSEC,
    SKYPATROL_THREADS,
)

LOGGER = logging.getLogger(__name__)
_SKYPATROL_CLIENT = None


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def get_asassn_lightcurve(
    asassn_id: str,
    ra: float | None = None,
    dec: float | None = None,
    cache_dir: Path = LIGHTCURVE_CACHE_DIR,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    retries: int = REQUEST_RETRIES,
    force: bool = False,
    source: str = "skypatrol",
) -> pd.DataFrame | None:
    """Fetch one ASAS-SN light curve with cache, timeout, and retries."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_safe_filename(asassn_id)}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    if source in {"skypatrol", "auto"} and ra is not None and dec is not None:
        skypatrol_lc = get_skypatrol_lightcurve(
            asassn_id=asassn_id,
            ra=ra,
            dec=dec,
            cache_path=cache_path,
        )
        if skypatrol_lc is not None:
            return skypatrol_lc
        if source == "skypatrol":
            return None

    url = f"https://asas-sn.osu.edu/variables/{asassn_id}/light_curve"
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 404:
                LOGGER.warning("Light curve endpoint returned 404 for %s; skipping retries for this target", asassn_id)
                return None
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("ASAS-SN response was not a JSON list")
            raw = pd.DataFrame(payload)
            cleaned = clean_lightcurve(raw)
            if cleaned is None:
                return None
            cleaned.to_csv(cache_path, index=False)
            return cleaned
        except Exception as exc:
            LOGGER.warning("Light curve fetch failed for %s on attempt %s/%s: %s", asassn_id, attempt, retries, exc)
            if attempt < retries:
                time.sleep(REQUEST_BACKOFF_SECONDS * attempt)
    return None


def get_skypatrol_client():
    """Create one Sky Patrol client lazily if pyasassn is installed/configured."""
    global _SKYPATROL_CLIENT
    if _SKYPATROL_CLIENT is not None:
        return _SKYPATROL_CLIENT
    try:
        from pyasassn.client import SkyPatrolClient
    except ImportError:
        LOGGER.warning(
            "pyasassn is not installed. Install it with: "
            "pip install git+https://github.com/asas-sn/skypatrol.git"
        )
        return None
    try:
        _SKYPATROL_CLIENT = SkyPatrolClient()
        return _SKYPATROL_CLIENT
    except Exception as exc:
        LOGGER.warning("Could not initialize Sky Patrol client: %s", exc)
        return None


def get_skypatrol_lightcurve(
    asassn_id: str,
    ra: float,
    dec: float,
    cache_path: Path,
    radius_arcsec: float = SKYPATROL_RADIUS_ARCSEC,
    threads: int = SKYPATROL_THREADS,
) -> pd.DataFrame | None:
    """Download a light curve from ASAS-SN Sky Patrol by coordinate cone search."""
    client = get_skypatrol_client()
    if client is None:
        return None
    try:
        lcs = client.cone_search(
            ra_deg=float(ra),
            dec_deg=float(dec),
            radius=float(radius_arcsec) / 3600.0,
            catalog="master_list",
            download=True,
            threads=threads,
        )
    except TypeError:
        # Some pyasassn versions use mode instead of download=True.
        try:
            query = (
                "SELECT asas_sn_id, ra_deg, dec_deg "
                "FROM master_list "
                f"WHERE DISTANCE(ra_deg, dec_deg, {float(ra)}, {float(dec)}) <= {float(radius_arcsec) / 3600.0}"
            )
            lcs = client.adql_query(query, mode="download_curves", threads=threads)
        except Exception as exc:
            LOGGER.warning("Sky Patrol query failed for %s: %s", asassn_id, exc)
            return None
    except Exception as exc:
        LOGGER.warning("Sky Patrol query failed for %s: %s", asassn_id, exc)
        return None

    raw = getattr(lcs, "data", None)
    if raw is None:
        if isinstance(lcs, pd.DataFrame):
            raw = lcs
        else:
            LOGGER.warning("Sky Patrol returned no light-curve data for %s", asassn_id)
            return None
    if raw.empty:
        return None

    cleaned = clean_lightcurve(raw)
    if cleaned is None:
        return None
    cleaned.to_csv(cache_path, index=False)
    return cleaned


def _choose_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def clean_lightcurve(df: pd.DataFrame) -> pd.DataFrame | None:
    """Standardize and clean ASAS-SN-like light curves."""
    if df is None or df.empty:
        return None

    time_col = _choose_column(df, ("HJD", "hjd", "JD", "jd", "MJD", "mjd"))
    mag_col = _choose_column(df, ("Vmag", "gmag", "mag", "Mag"))
    err_col = _choose_column(df, ("Verr", "gerr", "mag_err", "Magerr", "err"))
    if time_col is None or mag_col is None:
        raise ValueError("Light curve must contain time and magnitude columns")

    out = pd.DataFrame(
        {
            "time": pd.to_numeric(df[time_col], errors="coerce"),
            "mag": pd.to_numeric(df[mag_col], errors="coerce"),
        }
    )
    if err_col is not None:
        out["mag_err"] = pd.to_numeric(df[err_col], errors="coerce")
    else:
        out["mag_err"] = np.nan

    flag_col = _choose_column(df, ("quality", "Quality", "grade", "Grade", "flag", "Flag"))
    if flag_col is not None:
        valid_flag = ~df[flag_col].astype(str).str.lower().isin({"bad", "d", "f", "false", "1"})
        out = out.loc[valid_flag.to_numpy()]

    out = out.dropna(subset=["time", "mag"])
    out = out[out["mag"].between(5.0, 25.0)]
    if out["mag_err"].notna().any():
        out = out[(out["mag_err"].isna()) | (out["mag_err"].between(0.0, 1.0))]

    q1, q3 = out["mag"].quantile([0.25, 0.75])
    iqr = q3 - q1
    if np.isfinite(iqr) and iqr > 0:
        out = out[out["mag"].between(q1 - 3.0 * iqr, q3 + 3.0 * iqr)]

    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(out) < MIN_LIGHTCURVE_POINTS:
        return None
    return out
