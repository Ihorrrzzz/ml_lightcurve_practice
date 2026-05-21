from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Allow running from project root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightcurve_ml.degradation import degrade_dataset
from lightcurve_ml.evaluation import (
    compute_model_metrics,
    compute_reliability_curve,
    dataset_summary_table,
    per_class_report_table,
    plot_reliability_curves,
    save_confusion_matrix,
    save_latex_summary,
)
from lightcurve_ml.features import extract_features
from lightcurve_ml.plotting import plot_metric_vs_npoints
from lightcurve_ml.real_data import load_real_lightcurves_from_csv


RANDOM_STATE = 42
DEGRADED_POINTS = [5, 10, 20, 50]
REAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "real_lightcurves.csv"


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


def missing_input_message() -> str:
    return (
        f"Missing real-data CSV: {REAL_DATA_PATH}\n"
        "Create a normalized CSV at data/processed/real_lightcurves.csv with one observation per row "
        "and exactly these required columns:\n"
        "object_id,label,time,mag,mag_err,band\n"
        "Rows should share object_id and label across observations of the same source. "
        "time, mag, and mag_err must be numeric or convertible to numeric."
    )


def main() -> None:
    if not REAL_DATA_PATH.exists():
        raise SystemExit(missing_input_message())

    out_fig = PROJECT_ROOT / "outputs" / "figures"
    out_tab = PROJECT_ROOT / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading real light curves...")
    lightcurves = load_real_lightcurves_from_csv(REAL_DATA_PATH)
    if lightcurves.empty:
        raise SystemExit("No usable rows remain after cleaning real_lightcurves.csv.")
    dataset_summary_table(lightcurves, "real").to_csv(
        out_tab / "real_dataset_summary.csv",
        index=False,
    )

    print("[2/6] Splitting by object...")
    try:
        train_lc, test_lc = split_by_object(lightcurves)
    except ValueError as exc:
        raise SystemExit(
            "Could not create a stratified object-level train/test split. "
            "Check that each label has enough distinct object_id values."
        ) from exc

    print("[3/6] Extracting features from full training curves...")
    train_full_features = extract_features(train_lc)
    X_train_full, y_train_full = prepare_xy(train_full_features)

    print("[4/6] Training baseline model on full curves...")
    baseline = make_model(RANDOM_STATE)
    baseline.fit(X_train_full, y_train_full)

    print("[5/6] Creating degraded training set and training robust model...")
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

    print("[6/6] Evaluating both models on degraded test curves and saving outputs...")
    metric_rows = []
    per_class_rows = []
    reliability_curves = {}
    labels = sorted(train_full_features["label"].unique())

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

            metric_rows.append(compute_model_metrics(model_name, n, y_test, pred, proba, model.classes_))
            per_class_rows.append(
                per_class_report_table(
                    dataset_name="real",
                    model_name=model_name,
                    n_points=n,
                    y_true=y_test,
                    y_pred=pred,
                    labels=labels,
                )
            )

            save_confusion_matrix(
                y_test,
                pred,
                labels,
                title=f"Real-data confusion matrix: {model_name}, n={n}",
                out_path=out_fig / f"real_confusion_{model_name}_{n}.png",
            )

            if n == 10:
                reliability_curves[model_name] = compute_reliability_curve(y_test, pred, proba)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_tab / "real_metrics.csv", index=False)
    pd.concat(per_class_rows, ignore_index=True).to_csv(
        out_tab / "real_per_class_metrics.csv",
        index=False,
    )

    summary = metrics[[
        "model",
        "n_points",
        "accuracy",
        "f1_macro",
        "log_loss",
        "brier_score",
        "abstention_rate",
        "confident_accuracy",
    ]].copy()
    summary.to_csv(out_tab / "real_summary.csv", index=False)
    save_latex_summary(summary, out_tab / "real_latex_summary.tex")

    plot_metric_vs_npoints(summary, "accuracy", out_fig / "real_accuracy_vs_npoints.png")
    plot_metric_vs_npoints(summary, "f1_macro", out_fig / "real_f1_vs_npoints.png")
    plot_reliability_curves(
        reliability_curves,
        out_fig / "real_reliability_n10.png",
        title="Real-data reliability curve, n=10",
    )

    print("\nDone.")
    print(f"Tables saved to: {out_tab}")
    print(f"Figures saved to: {out_fig}")
    print("\nOpen outputs/tables/real_summary.csv first.")


if __name__ == "__main__":
    main()
