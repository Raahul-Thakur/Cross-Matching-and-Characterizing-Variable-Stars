"""Light-curve feature engineering for variable-star classification."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.stats import skew
from tqdm.auto import tqdm

from config import ALIAS_PERIODS_DAYS, ALIAS_TOLERANCE_DAYS, MAX_PERIOD_DAYS, MIN_LIGHTCURVE_POINTS, MIN_PERIOD_DAYS
from lightcurves import get_asassn_lightcurve
from utils import finite_or_nan

LOGGER = logging.getLogger(__name__)

IRREGULAR_LABEL_HINTS = ("M", "L", "SR", "IRR", "YSO", "AGN", "QSO", "VAR")


def is_periodic_label(label: str) -> bool:
    value = str(label).upper()
    return not any(hint in value for hint in IRREGULAR_LABEL_HINTS)


def eta_index(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    variance = np.var(values, ddof=1)
    if variance == 0:
        return np.nan
    return np.sum(np.diff(values) ** 2) / ((len(values) - 1) * variance)


def stetson_j(mag: np.ndarray, err: np.ndarray | None = None) -> float:
    if len(mag) < 2:
        return np.nan
    if err is None or np.all(~np.isfinite(err)):
        err = np.full_like(mag, np.nanstd(mag) if np.nanstd(mag) > 0 else 1.0)
    valid = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if valid.sum() < 2:
        return np.nan
    delta = np.sqrt(valid.sum() / (valid.sum() - 1)) * (mag[valid] - np.nanmean(mag[valid])) / err[valid]
    products = delta[:-1] * delta[1:]
    return np.nanmean(np.sign(products) * np.sqrt(np.abs(products)))


def estimate_period(lc: pd.DataFrame) -> tuple[float, float, bool]:
    """Estimate period using bounded Lomb-Scargle and flag common aliases."""
    time = lc["time"].to_numpy(dtype=float)
    mag = lc["mag"].to_numpy(dtype=float)
    err = lc["mag_err"].to_numpy(dtype=float) if "mag_err" in lc.columns else None
    valid = np.isfinite(time) & np.isfinite(mag)
    if err is not None and np.isfinite(err).any():
        valid &= (np.isfinite(err) & (err > 0))
        dy = err[valid]
    else:
        dy = None
    if valid.sum() < MIN_LIGHTCURVE_POINTS:
        return np.nan, np.nan, False

    frequency, power = LombScargle(time[valid], mag[valid], dy=dy).autopower(
        minimum_frequency=1.0 / MAX_PERIOD_DAYS,
        maximum_frequency=1.0 / MIN_PERIOD_DAYS,
        samples_per_peak=10,
    )
    if len(power) == 0:
        return np.nan, np.nan, False
    best_period = 1.0 / frequency[int(np.nanargmax(power))]
    best_power = float(np.nanmax(power))
    is_alias = any(abs(best_period - alias) <= ALIAS_TOLERANCE_DAYS for alias in ALIAS_PERIODS_DAYS)
    return finite_or_nan(best_period), best_power, is_alias


def choose_analysis_period(
    lomb_scargle_period: float,
    catalog_period: float,
    period_alias_flag: bool,
) -> tuple[float, float, str]:
    """Choose a defensible analysis period for plots/features.

    Ground-based surveys often produce one-day aliases. ASAS-SN catalog periods
    are already vetted, so use them when our simple Lomb-Scargle period is an
    obvious alias or far from the catalog period.
    """
    has_ls = np.isfinite(lomb_scargle_period) and lomb_scargle_period > 0
    has_catalog = np.isfinite(catalog_period) and catalog_period > 0
    if not has_ls and has_catalog:
        return catalog_period, np.nan, "catalog_no_lomb_scargle"
    if has_ls and not has_catalog:
        return lomb_scargle_period, np.nan, "lomb_scargle_no_catalog"
    if not has_ls and not has_catalog:
        return np.nan, np.nan, "no_period"

    ratio = lomb_scargle_period / catalog_period
    catalog_is_alias = any(abs(catalog_period - alias) <= ALIAS_TOLERANCE_DAYS for alias in ALIAS_PERIODS_DAYS)
    harmonic_match = any(abs(ratio - harmonic) <= 0.05 for harmonic in (0.5, 1.0, 2.0))
    if period_alias_flag and not catalog_is_alias:
        return catalog_period, ratio, "catalog_replaced_daily_alias"
    if not harmonic_match and (ratio < 0.2 or ratio > 5.0):
        return catalog_period, ratio, "catalog_replaced_large_disagreement"
    return lomb_scargle_period, ratio, "lomb_scargle"


def compute_lc_features(lc: pd.DataFrame, catalog_period: float | None = None, label: str | None = None) -> dict[str, float] | None:
    if lc is None or len(lc) < MIN_LIGHTCURVE_POINTS:
        return None

    mag = lc["mag"].to_numpy(dtype=float)
    err = lc["mag_err"].to_numpy(dtype=float) if "mag_err" in lc.columns else None
    lomb_scargle_period, period_power, period_alias_flag = estimate_period(lc)
    robust_amp = np.nanpercentile(mag, 95) - np.nanpercentile(mag, 5)
    max_min_amp = np.nanmax(mag) - np.nanmin(mag)
    catalog_period = pd.to_numeric(catalog_period, errors="coerce")
    period, period_ratio, period_source = choose_analysis_period(lomb_scargle_period, catalog_period, period_alias_flag)

    return {
        "period": period,
        "lomb_scargle_period": lomb_scargle_period,
        "period_source": period_source,
        "period_power": period_power,
        "period_alias_flag": bool(period_alias_flag),
        "catalog_period": finite_or_nan(catalog_period),
        "period_catalog_ratio": finite_or_nan(period_ratio),
        "amplitude_p95_p05": finite_or_nan(robust_amp),
        "amplitude_max_min": finite_or_nan(max_min_amp),
        "median_mag": finite_or_nan(np.nanmedian(mag)),
        "std_mag": finite_or_nan(np.nanstd(mag, ddof=1)),
        "iqr_mag": finite_or_nan(np.nanpercentile(mag, 75) - np.nanpercentile(mag, 25)),
        "skewness": finite_or_nan(skew(mag, nan_policy="omit")),
        "eta_index": finite_or_nan(eta_index(mag)),
        "stetson_j": finite_or_nan(stetson_j(mag, err)),
        "n_points": int(len(lc)),
        "is_periodic_label": bool(is_periodic_label(label or "")),
        "feature_source": "lightcurve",
    }


def compute_catalog_fallback_features(row: pd.Series) -> dict[str, float] | None:
    """Use catalog period/amplitude when ASAS-SN light-curve download is unavailable."""
    period = pd.to_numeric(row.get("asassn_period_asassn", np.nan), errors="coerce")
    amplitude = pd.to_numeric(row.get("asassn_amplitude_asassn", np.nan), errors="coerce")
    median_mag = pd.to_numeric(row.get("asassn_asassn_mag", np.nan), errors="coerce")
    if not np.isfinite(period) and not np.isfinite(amplitude) and not np.isfinite(median_mag):
        return None
    return {
        "period": finite_or_nan(period),
        "lomb_scargle_period": np.nan,
        "period_source": "catalog_only",
        "period_power": np.nan,
        "period_alias_flag": bool(np.isfinite(period) and any(abs(period - alias) <= ALIAS_TOLERANCE_DAYS for alias in ALIAS_PERIODS_DAYS)),
        "catalog_period": finite_or_nan(period),
        "period_catalog_ratio": 1.0 if np.isfinite(period) and period > 0 else np.nan,
        "amplitude_p95_p05": finite_or_nan(amplitude),
        "amplitude_max_min": finite_or_nan(amplitude),
        "median_mag": finite_or_nan(median_mag),
        "std_mag": np.nan,
        "iqr_mag": np.nan,
        "skewness": np.nan,
        "eta_index": np.nan,
        "stetson_j": np.nan,
        "n_points": 0,
        "is_periodic_label": bool(is_periodic_label(row.get("asassn_class", ""))),
        "feature_source": "catalog",
    }


def build_feature_table(
    cross_df: pd.DataFrame,
    max_lightcurves: int | None = None,
    download_lightcurves: bool = True,
    lightcurve_source: str = "skypatrol",
) -> pd.DataFrame:
    """Download/cache light curves and assemble features merged with cross-match data."""
    if cross_df.empty:
        return pd.DataFrame()
    rows = []
    sample = cross_df.head(max_lightcurves) if max_lightcurves else cross_df
    iterator = tqdm(sample.iterrows(), total=len(sample), desc="Light curves")
    for _, row in iterator:
        asassn_id = row["asassn_asassn_id"]
        lc = (
            get_asassn_lightcurve(
                asassn_id,
                ra=row.get("gaia_ra", row.get("asassn_ra")),
                dec=row.get("gaia_dec", row.get("asassn_dec")),
                source=lightcurve_source,
            )
            if download_lightcurves
            else None
        )
        if lc is None:
            features = compute_catalog_fallback_features(row)
        else:
            features = compute_lc_features(
                lc,
                catalog_period=row.get("asassn_period_asassn", np.nan),
                label=row.get("asassn_class", ""),
            )
        if features is None:
            continue
        rows.append(
            {
                "source_id": row["gaia_source_id"],
                "asassn_id": asassn_id,
                "label": row["asassn_class"],
                "ra": row["gaia_ra"],
                "dec": row["gaia_dec"],
                "l": row.get("gaia_l", np.nan),
                "b": row.get("gaia_b", np.nan),
                "bp_rp": row["gaia_bp_rp"],
                "phot_g_mean_mag": row["gaia_phot_g_mean_mag"],
                "abs_mag_g": row["gaia_abs_mag_g"],
                "parallax": row["gaia_parallax"],
                "parallax_error": row["gaia_parallax_error"],
                "parallax_over_error": row["gaia_parallax_over_error"],
                "ruwe": row["gaia_ruwe"],
                "separation_arcsec": row["separation_arcsec"],
                "crossmatch_quality_ok": row["crossmatch_quality_ok"],
                "variability_group": "periodic" if features["is_periodic_label"] else "irregular",
                **features,
            }
        )
    return pd.DataFrame(rows)
