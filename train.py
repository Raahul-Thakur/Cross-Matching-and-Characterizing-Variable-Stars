"""Model training, validation, and persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import CV_FOLDS, MODEL_DIR, OUTPUT_DIR, RANDOM_SEED, RARE_CLASS_MIN_SAMPLES, TEST_SIZE

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "period",
    "period_power",
    "amplitude_p95_p05",
    "median_mag",
    "std_mag",
    "iqr_mag",
    "skewness",
    "eta_index",
    "stetson_j",
    "n_points",
    "bp_rp",
    "abs_mag_g",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "ruwe",
    "separation_arcsec",
]


def prepare_training_data(df: pd.DataFrame, min_samples: int = RARE_CLASS_MIN_SAMPLES) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Drop unusable rows and merge rare labels into OTHER."""
    required = set(FEATURE_COLUMNS + ["label"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {sorted(missing)}")
    clean = df.copy()
    clean = clean[clean["crossmatch_quality_ok"].fillna(True)]
    clean = clean.dropna(subset=["label"])
    counts = clean["label"].value_counts()
    rare = counts[counts < min_samples].index
    clean["label_model"] = clean["label"].where(~clean["label"].isin(rare), "OTHER")
    clean = clean[clean["label_model"].notna()]
    X = clean[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    usable_feature_columns = X.columns[X.notna().any(axis=0)].tolist()
    dropped = sorted(set(FEATURE_COLUMNS) - set(usable_feature_columns))
    if dropped:
        LOGGER.warning("Dropping all-empty feature columns: %s", dropped)
    X = X[usable_feature_columns]
    y = clean["label_model"].astype(str)
    return X, y, clean, usable_feature_columns


def train_and_evaluate(
    feat_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    tune: bool = True,
) -> dict:
    """Train baseline and Random Forest models with CV and held-out metrics."""
    X, y, clean, feature_columns = prepare_training_data(feat_df)
    if len(clean) < 10 or y.nunique() < 2:
        raise ValueError("Need at least 10 feature rows and 2 classes to train a useful model")

    class_counts = y.value_counts()
    n_splits = min(CV_FOLDS, int(class_counts.min()))
    use_cv = n_splits >= 2

    indices = clean.index.to_numpy()
    stratify = y if class_counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    train_class_counts = y_train.value_counts()
    train_n_splits = min(n_splits, int(train_class_counts.min()))
    use_train_cv = train_n_splits >= 2

    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced", n_jobs=1)),
        ]
    )

    if tune and use_train_cv:
        search = RandomizedSearchCV(
            rf,
            param_distributions={
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            n_iter=12,
            scoring="f1_macro",
            cv=StratifiedKFold(n_splits=train_n_splits, shuffle=True, random_state=RANDOM_SEED),
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = rf.set_params(model__n_estimators=300, model__max_depth=12)
        model.fit(X_train, y_train)
        best_params = model.get_params()

    y_pred = model.predict(X_test)
    labels = sorted(y.unique())
    metrics = {
        "class_balance": class_counts.to_dict(),
        "feature_columns_used": feature_columns,
        "feature_columns_dropped": sorted(set(FEATURE_COLUMNS) - set(feature_columns)),
        "baseline": {
            "balanced_accuracy": balanced_accuracy_score(y_test, baseline_pred),
            "macro_f1": f1_score(y_test, baseline_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, baseline_pred, average="weighted", zero_division=0),
        },
        "random_forest": {
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
            "labels": labels,
            "best_params": best_params,
        },
    }

    if use_cv:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        cv_pred = cross_val_predict(model, X, y, cv=cv)
        metrics["cross_validation"] = {
            "folds": n_splits,
            "balanced_accuracy": balanced_accuracy_score(y, cv_pred),
            "macro_f1": f1_score(y, cv_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, cv_pred, average="weighted", zero_division=0),
        }

    result = permutation_importance(model, X_test, y_test, scoring="f1_macro", random_state=RANDOM_SEED, n_repeats=10)
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "random_forest_variable_classifier.joblib")
    clean.to_csv(output_dir / "training_feature_table.csv", index=False)
    pd.DataFrame({"train_index": train_idx}).to_csv(output_dir / "train_indices.csv", index=False)
    pd.DataFrame({"test_index": test_idx}).to_csv(output_dir / "test_indices.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return {"model": model, "metrics": metrics, "features": clean, "feature_importance": importance_df}
