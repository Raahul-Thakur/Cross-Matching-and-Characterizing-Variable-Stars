"""Offline sanity check for the pipeline's pure-Python pieces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crossmatch import crossmatch_gaia_asassn
from data_query import prepare_gaia_table
from features import compute_lc_features
from train import prepare_training_data
from utils import ensure_directories


def synthetic_lightcurve(period: float = 0.75, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    time = np.sort(rng.uniform(0, 30, n))
    mag = 13.0 + 0.25 * np.sin(2 * np.pi * time / period) + rng.normal(0, 0.03, n)
    return pd.DataFrame({"time": time, "mag": mag, "mag_err": np.full(n, 0.03)})


def main() -> None:
    ensure_directories()
    gaia = prepare_gaia_table(
        pd.DataFrame(
            {
                "source_id": [1, 2],
                "ra": [10.0, 20.0],
                "dec": [-5.0, 1.0],
                "l": [120.0, 130.0],
                "b": [5.0, 10.0],
                "parallax": [2.0, 1.5],
                "parallax_error": [0.1, 0.1],
                "phot_g_mean_mag": [13.0, 14.0],
                "bp_rp": [1.2, 0.8],
                "ruwe": [1.1, 1.2],
                "ag_gspphot": [0.1, 0.0],
            }
        )
    )
    asassn = pd.DataFrame(
        {
            "asassn_id": ["ASASSN-V J0001", "ASASSN-V J0002"],
            "ra": [10.0002, 20.0002],
            "dec": [-5.0002, 1.0002],
            "class": ["RRAB", "EA"],
            "period_asassn": [0.75, 1.2],
            "amplitude_asassn": [0.5, 0.4],
            "asassn_mag": [13.1, 14.1],
        }
    )
    cross = crossmatch_gaia_asassn(gaia, asassn, radius_arcsec=2.0)
    features = compute_lc_features(synthetic_lightcurve(), catalog_period=0.75, label="RRAB")
    assert not cross.empty
    assert features is not None
    assert abs(gaia.loc[0, "abs_mag_g"] - (13.0 + 5 * np.log10(2.0) - 10 - 0.1)) < 1e-9

    train_df = pd.DataFrame(
        [
            {
                "label": "RRAB",
                "crossmatch_quality_ok": True,
                "period": features["period"],
                "period_power": features["period_power"],
                "amplitude_p95_p05": features["amplitude_p95_p05"],
                "median_mag": features["median_mag"],
                "std_mag": features["std_mag"],
                "iqr_mag": features["iqr_mag"],
                "skewness": features["skewness"],
                "eta_index": features["eta_index"],
                "stetson_j": features["stetson_j"],
                "n_points": features["n_points"],
                "bp_rp": 1.2,
                "abs_mag_g": gaia.loc[0, "abs_mag_g"],
                "parallax": 2.0,
                "parallax_error": 0.1,
                "parallax_over_error": 20.0,
                "ruwe": 1.1,
                "separation_arcsec": cross.loc[0, "separation_arcsec"],
            }
        ]
    )
    X, y, clean, feature_columns = prepare_training_data(train_df, min_samples=1)
    assert len(X) == len(y) == len(clean) == 1
    assert feature_columns
    print("Smoke test passed")


if __name__ == "__main__":
    main()
