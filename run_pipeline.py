"""Run the Gaia/ASAS-SN variable-star pipeline end to end."""

from __future__ import annotations

import argparse
import logging

from pathlib import Path

from config import ASASSN_CATALOG_PATH, CROSSMATCH_RADIUS_ARCSEC, INTERIM_DIR, MAX_LIGHTCURVES, OUTPUT_DIR, PLOT_DIR, RAW_DIR
from crossmatch import crossmatch_gaia_asassn, load_asassn_catalog
from data_query import load_gaia_table, query_gaia_variables
from features import build_feature_table
from plots import (
    plot_class_distribution,
    plot_feature_correlation,
    plot_hr_diagram,
    plot_period_amplitude,
    plot_phase_folded_examples,
    plot_sky_distribution,
)
from train import train_and_evaluate
from utils import ensure_directories, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-gaia", action="store_true", help="Re-query Gaia instead of using the cached Gaia table.")
    parser.add_argument("--sample-size", type=int, default=None, help="Gaia sample size for a live query.")
    parser.add_argument("--max-lightcurves", type=int, default=MAX_LIGHTCURVES, help="Maximum matched light curves to process.")
    parser.add_argument("--radius-arcsec", type=float, default=CROSSMATCH_RADIUS_ARCSEC, help="Cross-match radius in arcsec.")
    parser.add_argument(
        "--asassn-catalog",
        type=Path,
        default=ASASSN_CATALOG_PATH,
        help="Path to a downloaded ASAS-SN variable-star catalog CSV.",
    )
    parser.add_argument(
        "--use-test-data",
        action="store_true",
        help="Use the tiny bundled Gaia/ASAS-SN fixture to verify the pipeline wiring.",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Do not download light curves; use ASAS-SN catalog period/amplitude/magnitude features only.",
    )
    parser.add_argument(
        "--lightcurve-source",
        choices=["skypatrol", "legacy", "auto"],
        default="skypatrol",
        help="Light-curve backend when not using --catalog-only. 'skypatrol' uses RA/Dec cone searches.",
    )
    parser.add_argument("--skip-training", action="store_true", help="Build features and plots but do not train a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    setup_logging()

    if args.use_test_data:
        LOGGER.warning("Using tiny bundled test data. This is only for checking that the code runs.")
        gaia_df = load_gaia_table(Path("data/test/gaia_sample.csv"))
        asassn_catalog_path = Path("data/test/asassn_sample.csv")
    else:
        gaia_path = RAW_DIR / "gaia_dr3_variables.csv"
        if gaia_path.exists() and not args.force_gaia:
            gaia_df = load_gaia_table(gaia_path)
        else:
            gaia_df = query_gaia_variables(sample_size=args.sample_size or 20000, force=args.force_gaia)
        asassn_catalog_path = args.asassn_catalog

    try:
        asassn_df = load_asassn_catalog(asassn_catalog_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            "\nASAS-SN catalog CSV is missing.\n\n"
            f"Expected path:\n  {asassn_catalog_path.resolve()}\n\n"
            "What to do:\n"
            "  1. Download/export the ASAS-SN Variable Stars catalog as CSV.\n"
            "  2. Save it as data/raw/asassn_catalog.csv, or pass its path with:\n"
            "     python CMCVA.py --asassn-catalog C:\\path\\to\\asassn_catalog.csv\n\n"
            "To only verify that the code wiring works, run:\n"
            "  python CMCVA.py --use-test-data --skip-training\n"
        ) from exc
    cross_df = crossmatch_gaia_asassn(gaia_df, asassn_df, radius_arcsec=args.radius_arcsec)
    if cross_df.empty:
        LOGGER.warning("No cross-matches found; stopping before light-curve download.")
        return

    feature_path = INTERIM_DIR / "variable_star_features.csv"
    feat_df = build_feature_table(
        cross_df,
        max_lightcurves=args.max_lightcurves,
        download_lightcurves=not args.catalog_only,
        lightcurve_source=args.lightcurve_source,
    )
    if feat_df.empty:
        LOGGER.warning("No usable light-curve features were built.")
        return
    feat_df.to_csv(feature_path, index=False)
    feat_df.to_csv(OUTPUT_DIR / "variable_star_features.csv", index=False)

    write_json(
        OUTPUT_DIR / "feature_generation_metadata.json",
        {
            "crossmatch_radius_arcsec": args.radius_arcsec,
            "max_lightcurves": args.max_lightcurves,
            "feature_table": str(feature_path),
            "feature_mode": "catalog_only" if args.catalog_only else "lightcurve_with_catalog_fallback",
            "lightcurve_source": args.lightcurve_source,
            "uses_catalog_period_as_training_feature": False,
            "absolute_magnitude_formula": "M_G = G + 5 * log10(parallax_mas) - 10 - A_G",
        },
    )

    plot_sky_distribution(feat_df, PLOT_DIR)
    plot_class_distribution(feat_df, PLOT_DIR)
    plot_period_amplitude(feat_df, PLOT_DIR)
    plot_feature_correlation(feat_df, PLOT_DIR)
    plot_hr_diagram(feat_df, PLOT_DIR)
    plot_phase_folded_examples(feat_df, PLOT_DIR)

    if not args.skip_training:
        train_and_evaluate(feat_df)

    LOGGER.info("Pipeline complete. Outputs are in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
