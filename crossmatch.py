"""Batch/vectorized cross-matching between Gaia and ASAS-SN catalogs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

from config import ASASSN_CATALOG_PATH, CROSSMATCH_RADIUS_ARCSEC, INTERIM_DIR

LOGGER = logging.getLogger(__name__)

ASASSN_COLUMN_ALIASES = {
    "asassn_id": ["asassn_id", "ASASSN_NAME", "ASASSN-V", "asassn_name", "asas_sn_id", "ID", "name"],
    "ra": ["ra", "RA", "RAdeg", "raj2000", "RAJ2000"],
    "dec": ["dec", "DEC", "DEdeg", "dej2000", "DEJ2000"],
    "class": ["class", "CLASS", "variable_type", "Type"],
    "period_asassn": ["period_asassn", "PERIOD", "Period", "Per"],
    "amplitude_asassn": ["amplitude_asassn", "AMPLITUDE", "Amplitude", "Amp"],
    "asassn_mag": ["asassn_mag", "Vmag", "Gmag", "mean_vmag", "mean_mag", "MeanMag"],
}


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def standardize_asassn_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common ASAS-SN catalog column names."""
    rename = {}
    for canonical, aliases in ASASSN_COLUMN_ALIASES.items():
        found = _first_existing_column(df, aliases)
        if found is not None:
            rename[found] = canonical
    out = df.rename(columns=rename).copy()
    required = {"asassn_id", "ra", "dec", "class"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"ASAS-SN catalog missing required columns: {sorted(missing)}")
    for col in ("ra", "dec", "period_asassn", "amplitude_asassn", "asassn_mag"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["ra", "dec"]).drop_duplicates("asassn_id").reset_index(drop=True)


def load_asassn_catalog(path: Path = ASASSN_CATALOG_PATH) -> pd.DataFrame:
    """Load a local ASAS-SN catalog file for vectorized cross-matching."""
    if not path.exists():
        raise FileNotFoundError(
            f"ASAS-SN catalog not found at {path}. Place a downloaded catalog there, "
            "or use the provided test fixture for offline checks."
        )
    LOGGER.info("Loading ASAS-SN catalog from %s", path)
    return standardize_asassn_catalog(read_vizier_table(path))


def read_vizier_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/semicolon VizieR ASU exports and remove unit/rule rows."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_data_line = ""
        skiprows = 0
        for line in handle:
            if line.startswith("#") or not line.strip():
                skiprows += 1
                continue
            first_data_line = line
            break

    if not first_data_line:
        raise ValueError(f"No tabular data found in {path}")

    delimiter_counts = {
        ";": first_data_line.count(";"),
        "\t": first_data_line.count("\t"),
        ",": first_data_line.count(","),
    }
    sep = max(delimiter_counts, key=delimiter_counts.get)
    df = pd.read_csv(path, sep=sep, comment="#", skiprows=skiprows, low_memory=False)
    df = df.dropna(how="all")

    if not df.empty:
        first_col = df.columns[0]
        first_values = df[first_col].astype(str).str.strip()
        unit_or_rule = first_values.eq("") | first_values.str.fullmatch("-+")
        df = df.loc[~unit_or_rule].reset_index(drop=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin({"", "nan", "--"}), col] = np.nan
    return df


def crossmatch_gaia_asassn(
    gaia_df: pd.DataFrame,
    asassn_df: pd.DataFrame,
    radius_arcsec: float = CROSSMATCH_RADIUS_ARCSEC,
    output_path: Path = INTERIM_DIR / "gaia_asassn_crossmatch.csv",
) -> pd.DataFrame:
    """Vectorized nearest-neighbor cross-match with duplicate handling."""
    if gaia_df.empty or asassn_df.empty:
        return pd.DataFrame()

    gaia_coord = SkyCoord(gaia_df["ra"].to_numpy() * u.deg, gaia_df["dec"].to_numpy() * u.deg)
    asassn_coord = SkyCoord(asassn_df["ra"].to_numpy() * u.deg, asassn_df["dec"].to_numpy() * u.deg)
    idx_gaia, idx_asassn, sep2d, _ = search_around_sky(
        gaia_coord,
        asassn_coord,
        seplimit=radius_arcsec * u.arcsec,
    )

    if len(idx_gaia) == 0:
        LOGGER.warning("No Gaia/ASAS-SN matches within %.2f arcsec", radius_arcsec)
        return pd.DataFrame()

    matches = gaia_df.iloc[idx_gaia].reset_index(drop=True).add_prefix("gaia_")
    catalog_matches = asassn_df.iloc[idx_asassn].reset_index(drop=True).add_prefix("asassn_")
    out = pd.concat([matches, catalog_matches], axis=1)
    out["separation_arcsec"] = sep2d.arcsec
    out = validate_crossmatch(out)
    out = (
        out.sort_values(["separation_arcsec", "gaia_parallax_over_error"], ascending=[True, False])
        .drop_duplicates("gaia_source_id")
        .drop_duplicates("asassn_asassn_id")
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    LOGGER.info("Saved %s unique cross-matches to %s", len(out), output_path)
    return out


def validate_crossmatch(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple position/magnitude/color checks without discarding useful audit data."""
    out = df.copy()
    out["position_match_ok"] = out["separation_arcsec"] <= CROSSMATCH_RADIUS_ARCSEC
    if "asassn_asassn_mag" in out.columns:
        out["gaia_asassn_mag_delta"] = np.abs(out["gaia_phot_g_mean_mag"] - out["asassn_asassn_mag"])
        out["magnitude_match_ok"] = out["gaia_asassn_mag_delta"] <= 2.0
    else:
        out["gaia_asassn_mag_delta"] = np.nan
        out["magnitude_match_ok"] = True
    out["color_match_ok"] = out["gaia_bp_rp"].between(-1.0, 6.0)
    out["crossmatch_quality_ok"] = out[["position_match_ok", "magnitude_match_ok", "color_match_ok"]].all(axis=1)
    return out
