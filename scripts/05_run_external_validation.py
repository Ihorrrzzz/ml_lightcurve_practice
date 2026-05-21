from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Allow running from project root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightcurve_ml.degradation import degrade_dataset
from lightcurve_ml.evaluation import (
    compute_model_metrics,
    dataset_summary_table,
    per_class_report_table,
    save_confusion_matrix,
    save_latex_table,
)
from lightcurve_ml.features import extract_features
from lightcurve_ml.real_data import load_real_lightcurves_from_csv


RANDOM_STATE = 42
DEGRADED_POINTS = [5, 10, 20, 50]
OGLE_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "real_lightcurves.csv"
EXTERNAL_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "external_test_lightcurves.csv"


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


def prepare_xy(features: pd.DataFrame):
    y = features["label"].to_numpy()
    X = features.drop(columns=["object_id", "label"])
    return X, y


def missing_external_message() -> str:
    return (
        f"Missing external test CSV: {EXTERNAL_TEST_PATH}\n"
        "Create this file in normalized format with one observation per row and columns:\n"
        "object_id,label,time,mag,mag_err,band\n"
        "The data can come from ASAS-SN, Gaia, BHTOM, or another independent labelled source."
    )


def evaluate_scenario(
    scenario: str,
    n_points: int | str,
    test_lc: pd.DataFrame,
    models: list[tuple[str, str, object]],
    out_fig: Path,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Evaluate all models on one external-data scenario."""
    test_features = extract_features(test_lc)
    X_test, y_test = prepare_xy(test_features)

    rows = []
    per_class = []
    external_labels = set(test_features["label"].unique())

    for model_name, train_regime, model in models:
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        metrics = compute_model_metrics(
            model_name=model_name,
            n_points=n_points,
            y_true=y_test,
            y_pred=pred,
            proba=proba,
            classes=model.classes_,
        )
        metrics["train_regime"] = train_regime
        metrics["scenario"] = scenario
        rows.append(metrics)

        labels = sorted(external_labels | set(model.classes_))
        class_table = per_class_report_table(
            dataset_name="external_validation",
            model_name=model_name,
            n_points=n_points,
            y_true=y_test,
            y_pred=pred,
            labels=labels,
        )
        class_table["scenario"] = scenario
        per_class.append(class_table)

        save_confusion_matrix(
            y_test,
            pred,
            labels,
            title=f"External validation: {model_name}, {scenario}",
            out_path=out_fig / f"external_validation_confusion_{model_name}_{scenario}.png",
        )

    return rows, per_class


def plot_external_metric(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Plot sparse external validation metrics as a function of retained points."""
    plot_data = summary[summary["scenario"].str.startswith("early_")].copy()
    plot_data["n_points_numeric"] = pd.to_numeric(plot_data["n_points"], errors="coerce")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for model_name, sub in plot_data.groupby("model"):
        sub = sub.sort_values("n_points_numeric")
        ax.plot(sub["n_points_numeric"], sub[metric], marker="o", linewidth=2.0, label=model_name)

    ax.set_xlabel("Number of retained external points")
    ax.set_ylabel(metric)
    ax.set_title(f"External validation {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    if not OGLE_TRAIN_PATH.exists():
        print("Run python scripts/03_download_ogle_lmc_sample.py first.")
        return

    if not EXTERNAL_TEST_PATH.exists():
        print(missing_external_message())
        return

    out_fig = PROJECT_ROOT / "outputs" / "figures"
    out_tab = PROJECT_ROOT / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading OGLE training data and external test data...")
    ogle_train_lc = load_real_lightcurves_from_csv(OGLE_TRAIN_PATH)
    external_lc = load_real_lightcurves_from_csv(EXTERNAL_TEST_PATH)
    if ogle_train_lc.empty:
        raise SystemExit("No usable rows remain after cleaning real_lightcurves.csv.")
    if external_lc.empty:
        raise SystemExit("No usable rows remain after cleaning external_test_lightcurves.csv.")

    dataset_summary = pd.concat(
        [
            dataset_summary_table(ogle_train_lc, "ogle_train"),
            dataset_summary_table(external_lc, "external_test"),
        ],
        ignore_index=True,
    )
    dataset_summary.to_csv(out_tab / "external_validation_dataset_summary.csv", index=False)

    print("[2/5] Training baseline model on full OGLE curves...")
    train_full_features = extract_features(ogle_train_lc)
    X_train_full, y_train_full = prepare_xy(train_full_features)

    baseline = make_model(RANDOM_STATE)
    baseline.fit(X_train_full, y_train_full)

    print("[3/5] Training robust model on random degraded OGLE curves...")
    robust_train_parts = []
    for n_points in DEGRADED_POINTS:
        robust_train_parts.append(
            degrade_dataset(
                ogle_train_lc,
                n_points=n_points,
                seed=RANDOM_STATE + n_points,
                mode="random",
                extra_noise_scale=0.03,
            )
        )
    robust_train_lc = pd.concat(robust_train_parts, ignore_index=True)
    robust_train_features = extract_features(robust_train_lc)
    X_train_robust, y_train_robust = prepare_xy(robust_train_features)

    robust = make_model(RANDOM_STATE + 1)
    robust.fit(X_train_robust, y_train_robust)

    print("[4/5] Evaluating full and sparse external scenarios...")
    models = [
        ("baseline_full_train", "ogle_full_train", baseline),
        ("robust_random_train", "ogle_random_degraded_train", robust),
    ]

    metric_rows = []
    per_class_rows = []

    rows, class_rows = evaluate_scenario(
        scenario="full_external",
        n_points="full",
        test_lc=external_lc,
        models=models,
        out_fig=out_fig,
    )
    metric_rows.extend(rows)
    per_class_rows.extend(class_rows)

    for n_points in DEGRADED_POINTS:
        scenario = f"early_{n_points}"
        sparse_external_lc = degrade_dataset(
            external_lc,
            n_points=n_points,
            seed=RANDOM_STATE + 2000 + n_points,
            mode="early",
            extra_noise_scale=0.0,
        )
        rows, class_rows = evaluate_scenario(
            scenario=scenario,
            n_points=n_points,
            test_lc=sparse_external_lc,
            models=models,
            out_fig=out_fig,
        )
        metric_rows.extend(rows)
        per_class_rows.extend(class_rows)

    summary = pd.DataFrame(metric_rows)
    summary = summary[
        [
            "model",
            "train_regime",
            "scenario",
            "n_points",
            "accuracy",
            "f1_macro",
            "log_loss",
            "brier_score",
            "abstention_rate",
            "confident_accuracy",
            "n_test_objects",
        ]
    ].sort_values(["scenario", "model"])
    summary.to_csv(out_tab / "external_validation_summary.csv", index=False)
    save_latex_table(summary, out_tab / "external_validation_summary.tex")

    per_class = pd.concat(per_class_rows, ignore_index=True)
    per_class = per_class[
        [
            "dataset_name",
            "scenario",
            "model",
            "n_points",
            "label",
            "precision",
            "recall",
            "f1",
            "support",
        ]
    ]
    per_class.to_csv(out_tab / "external_validation_per_class_metrics.csv", index=False)

    print("[5/5] Saving plots...")
    plot_external_metric(summary, "accuracy", out_fig / "external_validation_accuracy_vs_npoints.png")
    plot_external_metric(summary, "f1_macro", out_fig / "external_validation_f1_vs_npoints.png")

    print("\nDone.")
    print(f"Tables saved to: {out_tab}")
    print(f"Figures saved to: {out_fig}")
    print("\nOpen outputs/tables/external_validation_summary.csv first.")


if __name__ == "__main__":
    main()
