# Project Content

## 1. LinkedIn Post Content

I built an exploratory astronomy and machine-learning pipeline to cross-match and characterize variable stars using Gaia DR3 and ASAS-SN.

The problem statement was: can we take known variable-star candidates from Gaia, connect them with an external variability catalog, fetch real light-curve data, engineer meaningful features, and build a simple classifier to study variability types?

For the data, I used Gaia DR3 through `astroquery`, the ASAS-SN variable-star catalog from VizieR table `II/366/catv2021`, and ASAS-SN Sky Patrol for time-series light curves. The pipeline cross-matches Gaia and ASAS-SN sources by sky position, applies Gaia quality filters such as positive parallax, parallax-over-error, valid color/magnitude, and RUWE, then stores angular separation and match-quality checks.

For each matched source, the project can run in two modes: a fast catalog-only mode using ASAS-SN period, amplitude, and magnitude, or a light-curve mode that downloads Sky Patrol photometry. In light-curve mode, it computes features such as period, robust amplitude, median magnitude, scatter, skewness, eta index, Stetson J, and Gaia color/absolute magnitude.

The outputs include a corrected HR diagram, sky-distribution plots, class distributions, period-amplitude diagrams, feature correlations, phase-folded light curves, model metrics, feature importance, and a saved Random Forest classifier.

Important caveat: this is an exploratory project, not a publication-grade classifier. ASAS-SN classes are treated as reference labels, daily period aliases can occur, and catalog periods are used to replace obvious aliases when needed. Still, it demonstrates a complete scientific data workflow from catalog cross-matching to light-curve feature engineering and classification.

## 2. Script for My Reel

What if we could combine Gaia star data with real variable-star light curves and use machine learning to understand stellar variability?

That was the goal of this project.

I built a Python pipeline for cross-matching and characterizing variable stars using Gaia DR3 and ASAS-SN.

The problem statement was simple: take variable-star candidates from Gaia, match them with an external variable-star catalog, fetch light curves, extract useful features, and classify variability types.

The data sources were Gaia DR3, accessed with `astroquery`; the ASAS-SN variable-star catalog from VizieR, specifically table `II/366/catv2021`; and ASAS-SN Sky Patrol for actual time-series light-curve data.

The pipeline first queries Gaia and applies quality filters like positive parallax, good parallax-over-error, valid color and magnitude, and RUWE less than 1.4. Then it cross-matches those Gaia sources with ASAS-SN using sky coordinates and stores the angular separation for each match.

Next, it can either run in catalog-only mode or download real Sky Patrol light curves. For the light curves, it computes features like period, amplitude, median magnitude, scatter, skewness, eta index, Stetson J, and Gaia-based absolute magnitude.

The outputs include HR diagrams, sky maps, class-distribution plots, period-amplitude diagrams, feature-correlation plots, phase-folded light curves, and Random Forest classification metrics.

The main caveat is that this is exploratory, not publication-grade. ASAS-SN labels are used as reference labels, and ground-based surveys can produce daily period aliases. To handle that, the pipeline keeps the raw Lomb-Scargle period but replaces obvious aliases with the ASAS-SN catalog period.

Overall, this project demonstrates a complete astronomy data-science workflow: cross-matching, light-curve analysis, feature engineering, visualization, and machine learning.
