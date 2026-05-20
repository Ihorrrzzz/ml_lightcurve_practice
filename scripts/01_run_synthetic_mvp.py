from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Allow running from project root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightcurve_ml.degradation import degrade_dataset
from lightcurve_ml.features import extract_features
from lightcurve_ml.plotting import plot_confusion_matrix, plot_metric_vs_npoints
from lightcurve_ml.synthetic import generate_synthetic_lightcurves


RANDOM_STATE = 42
N_PER_CLASS = 70
FULL_POINTS = 120
DEGRADED_POINTS = [5, 10, 20, 50]


def make_model(random_state: int = RANDOM_STATE):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    )


def split_by_object(lightcurves: pd.DataFrame):
    objects = lightcurves[["object_id", "label"]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        objects["object_id"],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=objects["label"],
    )

    train = lightcurves[lightcurves["object_id"].isin(train_ids)].copy()
    test = lightcurves[lightcurves["object_id"].isin(test_ids)].copy()
    return train, test


def prepare_xy(features: pd.DataFrame):
    y = features["label"].to_numpy()
    drop_cols = ["object_id", "label"]
    X = features.drop(columns=drop_cols)
    return X, y


def safe_log_loss(y_true, proba, labels):
    try:
        return float(log_loss(y_true, proba, labels=labels))
    except Exception:
        return float("nan")


def main() -> None:
    out_fig = PROJECT_ROOT / "outputs" / "figures"
    out_tab = PROJECT_ROOT / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    print("[1/7] Generating synthetic light curves...")
    lightcurves = generate_synthetic_lightcurves(
        n_per_class=N_PER_CLASS,
        n_points=FULL_POINTS,
        seed=RANDOM_STATE,
    )
    lightcurves.to_csv(PROJECT_ROOT / "data" / "processed" / "synthetic_lightcurves.csv", index=False)

    print("[2/7] Splitting by object...")
    train_lc, test_lc = split_by_object(lightcurves)

    print("[3/7] Extracting features from full training curves...")
    train_full_features = extract_features(train_lc)
    X_train_full, y_train_full = prepare_xy(train_full_features)

    labels = sorted(train_full_features["label"].unique())

    print("[4/7] Training baseline model on full curves...")
    baseline = make_model(RANDOM_STATE)
    baseline.fit(X_train_full, y_train_full)

    print("[5/7] Creating degraded training set and training robust model...")
    robust_train_parts = []
    for n in DEGRADED_POINTS:
        robust_train_parts.append(
            degrade_dataset(
                train_lc,
                n_points=n,
                seed=RANDOM_STATE + n,
                mode="random",
                extra_noise_scale=0.03,
            )
        )
    robust_train_lc = pd.concat(robust_train_parts, ignore_index=True)
    robust_train_features = extract_features(robust_train_lc)
    X_train_robust, y_train_robust = prepare_xy(robust_train_features)

    robust = make_model(RANDOM_STATE + 1)
    robust.fit(X_train_robust, y_train_robust)

    print("[6/7] Evaluating both models on degraded test curves...")
    metric_rows = []
    confusion_for_n10 = {}

    for n in DEGRADED_POINTS:
        degraded_test = degrade_dataset(
            test_lc,
            n_points=n,
            seed=RANDOM_STATE + 100 + n,
            mode="random",
            extra_noise_scale=0.03,
        )
        test_features = extract_features(degraded_test)
        X_test, y_test = prepare_xy(test_features)

        for model_name, model in [("baseline_full_train", baseline), ("robust_degraded_train", robust)]:
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)

            acc = accuracy_score(y_test, pred)
            f1_macro = f1_score(y_test, pred, average="macro")
            ll = safe_log_loss(y_test, proba, labels=model.classes_)

            max_conf = np.max(proba, axis=1)
            threshold = 0.60
            confident_mask = max_conf >= threshold
            abstention_rate = 1.0 - float(np.mean(confident_mask))
            if np.any(confident_mask):
                confident_accuracy = accuracy_score(y_test[confident_mask], pred[confident_mask])
            else:
                confident_accuracy = float("nan")

            metric_rows.append(
                {
                    "model": model_name,
                    "n_points": n,
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "log_loss": ll,
                    "confidence_threshold": threshold,
                    "abstention_rate": abstention_rate,
                    "confident_accuracy": confident_accuracy,
                    "n_test_objects": len(y_test),
                }
            )

            if n == 10:
                confusion_for_n10[model_name] = confusion_matrix(y_test, pred, labels=labels)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_tab / "synthetic_metrics.csv", index=False)

    summary = metrics[[
        "model",
        "n_points",
        "accuracy",
        "f1_macro",
        "log_loss",
        "abstention_rate",
        "confident_accuracy",
    ]].copy()
    summary.to_csv(out_tab / "synthetic_summary.csv", index=False)

    print("[7/7] Saving plots...")
    plot_metric_vs_npoints(summary, "accuracy", out_fig / "accuracy_vs_npoints.png")
    plot_metric_vs_npoints(summary, "f1_macro", out_fig / "f1_vs_npoints.png")

    for model_name, cm in confusion_for_n10.items():
        plot_confusion_matrix(
            cm,
            labels,
            title=f"Confusion matrix: {model_name}, n=10",
            out_path=out_fig / f"confusion_{model_name}_n10.png",
        )

    print("\nDone.")
    print(f"Tables saved to: {out_tab}")
    print(f"Figures saved to: {out_fig}")
    print("\nOpen outputs/tables/synthetic_summary.csv first.")


if __name__ == "__main__":
    main()
