# Cross-Matching and Characterizing Variable Stars in Gaia

An exploratory astronomy/data-science pipeline for cross-matching **Gaia DR3** variable-star sources with the **ASAS-SN variable-star catalog**, fetching **ASAS-SN Sky Patrol** light curves, extracting variability features, training a baseline machine-learning classifier, and generating diagnostic plots.

This is intended as a reproducible portfolio/research-learning project, not a publication-grade claim of new classifications or discoveries.

## Scope

- Query Gaia DR3 variable sources joined to `gaia_source` quality columns.
- Apply Gaia quality filters: positive parallax, parallax-over-error, valid color/magnitude, and `ruwe < 1.4`.
- Cross-match Gaia and ASAS-SN in batch with `astropy.coordinates.search_around_sky`.
- Keep nearest unique matches and store angular separation plus quality checks.
- Clean Sky Patrol light curves, estimate bounded Lomb-Scargle periods, flag common aliases, and compute robust variability features.
- Use ASAS-SN catalog periods to replace obvious daily-alias Lomb-Scargle periods when needed, while preserving `lomb_scargle_period` and `period_source` for auditability.
- Train a baseline model and tuned Random Forest with class-balance checks, cross-validation where possible, and saved metrics.
- Produce reproducible outputs in `outputs/` and cache intermediate data in `data/`.

## Data Inputs

### Data Sources

- **Gaia DR3** via `astroquery.gaia`: positions, parallaxes, colors, magnitudes, RUWE, and extinction estimates.
- **ASAS-SN variable-star catalog** via VizieR table `II/366/catv2021`: source names, RA/Dec, periods, amplitudes, mean magnitudes, classes, and Gaia IDs.
- **ASAS-SN Sky Patrol** via optional `pyasassn`: time-series light curves for matched ASAS-SN sources.

ASAS-SN variability classes are treated as **reference labels** for supervised learning. The classifier is therefore learning to reproduce ASAS-SN-style labels for matched Gaia/ASAS-SN sources, not independently proving new labels.

The pipeline expects a downloaded ASAS-SN catalog at:

```text
data/raw/asassn_catalog.csv
```

The file should contain at least source name, RA, Dec, and class columns. Common names such as `ASASSN_NAME`, `RA`, `DEC`, `CLASS`, `PERIOD`, and `AMPLITUDE` are normalized automatically.

VizieR ASU exports such as `asu.tsv` are supported, including semicolon-separated exports with columns like `ASASSN-V`, `RAJ2000`, `DEJ2000`, `Vmag`, `Amp`, `Per`, and `Type`.

If your catalog is somewhere else, pass it directly:

```bash
python CMCVA.py --asassn-catalog C:\path\to\asassn_catalog.csv
```

Gaia results are cached to:

```text
data/raw/gaia_dr3_variables.csv
```

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run an offline sanity check:

```bash
python smoke_test.py
```

Run the command-line pipeline with bundled tiny fixture data:

```bash
python CMCVA.py --use-test-data --skip-training
```

### Catalog-Only Mode

Catalog-only mode is fastest. It uses ASAS-SN catalog period/amplitude/magnitude plus Gaia features, without downloading time-series light curves:

```bash
python CMCVA.py --asassn-catalog data/raw/asu.tsv --catalog-only --max-lightcurves 2000
```

### Sky Patrol Light-Curve Mode

Install the optional Sky Patrol client:

```bash
pip install -r requirements-skypatrol.txt
```

Run a small light-curve test:

```bash
python CMCVA.py --asassn-catalog data/raw/asu.tsv --lightcurve-source skypatrol --max-lightcurves 20 --skip-training
```

Run a fuller exploratory light-curve analysis:

```bash
python CMCVA.py --asassn-catalog data/raw/asu.tsv --lightcurve-source skypatrol --max-lightcurves 100
```

This keeps the VizieR ASAS-SN catalog as the source list, then uses each matched star's RA/Dec to find and download the corresponding Sky Patrol light curve.

## Outputs

- `data/interim/gaia_asassn_crossmatch.csv`
- `data/interim/variable_star_features.csv`
- `outputs/variable_star_features.csv`
- `outputs/training_feature_table.csv`
- `outputs/metrics.json`
- `outputs/feature_importance.csv`
- `outputs/models/random_forest_variable_classifier.joblib`
- `outputs/plots/*.png`

Example generated plots:

- `outputs/plots/sky_distribution_radec.png`
- `outputs/plots/sky_distribution_galactic.png`
- `outputs/plots/class_distribution.png`
- `outputs/plots/hr_diagram_corrected_abs_mag.png`
- `outputs/plots/period_amplitude.png`
- `outputs/plots/feature_correlation.png`
- `outputs/plots/phase_folded_*.png`

Generated files, large downloaded catalogs, cached light curves, trained models, and plot outputs are ignored by Git by default. Keep small examples or selected figures only if you intentionally add them.

## Scientific Notes and Limitations

Absolute magnitude is computed as:

```text
M_G = G + 5 * log10(parallax_mas) - 10 - A_G
```

where `A_G` uses Gaia `ag_gspphot` when available and falls back to zero otherwise. For publication-quality work, replace this with a consistent extinction/reddening treatment, preferably using a 3D dust map and uncertainty propagation.

ASAS-SN labels are used as supervised labels, so the classifier learns to reproduce ASAS-SN classes for matched Gaia sources. It should not be treated as an independent validation of Gaia classes without additional checks.

The period finder uses bounded Lomb-Scargle and flags common aliases near 0.5, 1, and 2 days. Manual inspection of phase-folded light curves is still important.

For light-curve runs, the table stores both `lomb_scargle_period` and the selected analysis `period`. If the simple Lomb-Scargle period is a likely daily alias or strongly disagrees with the ASAS-SN catalog period, the pipeline uses the catalog period for downstream plots/features and records the reason in `period_source`.

Main limitations:

- This is exploratory and scoped to available public data/services.
- Sky Patrol downloads can be slow and may not return a light curve for every catalog source.
- Daily aliases may remain despite catalog-assisted period replacement.
- Extinction correction uses Gaia `ag_gspphot` when available and assumes zero otherwise.
- ASAS-SN classes are reference labels, not independent truth labels.
- Model scores depend on sample size, class balance, cross-match quality, and light-curve availability.
- Rare variability classes are merged into `OTHER` for training stability.
