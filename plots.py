"""Visualization helpers for cross-match and variable-star analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import PLOT_DIR
from lightcurves import get_asassn_lightcurve

sns.set_theme(style="whitegrid", context="notebook")


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_sky_distribution(df: pd.DataFrame, output_dir: Path = PLOT_DIR) -> None:
    if df.empty:
        return
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x="ra", y="dec", hue="label" if "label" in df else None, s=12, alpha=0.55, linewidth=0)
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.title("Sky Distribution")
    _savefig(output_dir / "sky_distribution_radec.png")

    if {"l", "b"}.issubset(df.columns):
        plt.figure(figsize=(9, 5))
        sns.scatterplot(data=df, x="l", y="b", hue="label" if "label" in df else None, s=12, alpha=0.55, linewidth=0)
        plt.xlabel("Galactic longitude l (deg)")
        plt.ylabel("Galactic latitude b (deg)")
        plt.title("Sky Distribution in Galactic Coordinates")
        _savefig(output_dir / "sky_distribution_galactic.png")


def plot_class_distribution(df: pd.DataFrame, output_dir: Path = PLOT_DIR, filename: str = "class_distribution.png") -> None:
    if df.empty or "label" not in df.columns:
        return
    counts = df["label"].value_counts()
    plt.figure(figsize=(10, max(4, 0.25 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index, color="#4c78a8")
    plt.xlabel("Count")
    plt.ylabel("Class")
    plt.title("Variable Class Distribution")
    _savefig(output_dir / filename)


def plot_period_amplitude(df: pd.DataFrame, output_dir: Path = PLOT_DIR) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="period",
        y="amplitude_p95_p05",
        hue="label",
        s=24,
        alpha=0.6,
        linewidth=0,
    )
    plt.xscale("log")
    plt.xlabel("Estimated period (days)")
    plt.ylabel("Robust amplitude P95-P05 (mag)")
    plt.title("Period-Amplitude Diagram")
    _savefig(output_dir / "period_amplitude.png")


def plot_feature_correlation(df: pd.DataFrame, output_dir: Path = PLOT_DIR) -> None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return
    plt.figure(figsize=(11, 9))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, square=False)
    plt.title("Feature Correlation")
    _savefig(output_dir / "feature_correlation.png")


def plot_hr_diagram(df: pd.DataFrame, output_dir: Path = PLOT_DIR) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="bp_rp", y="abs_mag_g", hue="label", s=24, alpha=0.65, linewidth=0)
    plt.gca().invert_yaxis()
    plt.xlabel("BP - RP")
    plt.ylabel("Extinction-corrected absolute G magnitude")
    plt.title("HR Diagram with Variable Star Classes")
    _savefig(output_dir / "hr_diagram_corrected_abs_mag.png")


def plot_normalized_confusion_matrix(y_true, y_pred, labels, output_dir: Path = PLOT_DIR) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(max(6, 0.5 * len(labels)), max(5, 0.45 * len(labels))))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    _savefig(output_dir / "normalized_confusion_matrix.png")


def plot_phase_folded_examples(df: pd.DataFrame, output_dir: Path = PLOT_DIR, per_class: int = 1) -> None:
    if df.empty:
        return
    if "feature_source" in df.columns:
        df = df[df["feature_source"].eq("lightcurve")]
    if df.empty:
        return
    examples = df.dropna(subset=["period"]).sort_values("period_power", ascending=False).groupby("label").head(per_class)
    for _, row in examples.iterrows():
        lc = get_asassn_lightcurve(row["asassn_id"])
        if lc is None or not np.isfinite(row["period"]) or row["period"] <= 0:
            continue
        phase = (lc["time"] % row["period"]) / row["period"]
        plt.figure(figsize=(7, 4))
        plt.scatter(phase, lc["mag"], s=10, alpha=0.65)
        plt.scatter(phase + 1, lc["mag"], s=10, alpha=0.65)
        plt.gca().invert_yaxis()
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        plt.title(f"{row['label']} | {row['asassn_id']} | P={row['period']:.4g} d")
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in str(row["label"]))
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in str(row["asassn_id"]))
        _savefig(output_dir / f"phase_folded_{safe_label}_{safe_id}.png")
