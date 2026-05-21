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
from lightcurve_ml.evaluation import compute_model_metrics
from lightcurve_ml.features import extract_features
from lightcurve_ml.real_data import load_real_lightcurves_from_csv


RANDOM_STATE = 42
DEGRADED_POINTS = [5, 10, 20, 50]
OGLE_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "real_lightcurves.csv"
EXTERNAL_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "external_test_lightcurves.csv"

FEATURE_SET_ORDER = [
    "all_features",
    "no_absolute_mag",
    "no_survey_timing",
    "shape_period",
    "no_error_features",
]
SCENARIO_ORDER = ["full_external", "early_5", "early_10", "early_20", "early_50"]


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


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        col
        for col in features.select_dtypes(include="number").columns
        if col not in {"object_id", "label"}
    ]


def define_feature_sets(all_features: list[str]) -> dict[str, list[str]]:
    shape_period_candidates = [
        "std_mag",
        "amplitude_p95_p5",
        "amp_to_err",
        "skew_mag",
        "kurtosis_mag",
        "max_abs_slope",
        "ls_period",
        "ls_power",
    ]

    return {
        "all_features": list(all_features),
        "no_absolute_mag": [col for col in all_features if col not in {"mean_mag", "median_mag"}],
        "no_survey_timing": [
            col
            for col in all_features
            if col not in {"mean_mag", "median_mag", "time_span", "median_cadence", "n_points"}
        ],
        "shape_period": [col for col in shape_period_candidates if col in all_features],
        "no_error_features": [col for col in all_features if col not in {"median_err", "amp_to_err"}],
    }


def prepare_xy(features: pd.DataFrame, feature_columns: list[str]):
    X = features.loc[:, feature_columns].copy()
    y = features["label"].to_numpy()
    return X, y


def make_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, scenario), sub in summary.groupby(["feature_set", "scenario"]):
        by_model = sub.set_index("model")
        baseline = by_model.loc["baseline_full_train"]
        robust = by_model.loc["robust_random_train"]
        rows.append(
            {
                "feature_set": feature_set,
                "scenario": scenario,
                "n_points": baseline["n_points"],
                "baseline_accuracy": float(baseline["accuracy"]),
                "robust_accuracy": float(robust["accuracy"]),
                "accuracy_delta": float(robust["accuracy"] - baseline["accuracy"]),
                "baseline_f1_macro": float(baseline["f1_macro"]),
                "robust_f1_macro": float(robust["f1_macro"]),
                "f1_delta": float(robust["f1_macro"] - baseline["f1_macro"]),
            }
        )

    delta = pd.DataFrame(rows)
    delta["feature_set"] = pd.Categorical(delta["feature_set"], categories=FEATURE_SET_ORDER, ordered=True)
    delta["scenario"] = pd.Categorical(delta["scenario"], categories=SCENARIO_ORDER, ordered=True)
    return delta.sort_values(["feature_set", "scenario"]).reset_index(drop=True)


def save_feature_sets(feature_sets: dict[str, list[str]], out_path: Path) -> None:
    rows = []
    for feature_set in FEATURE_SET_ORDER:
        for feature_name in feature_sets[feature_set]:
            rows.append({"feature_set": feature_set, "feature_name": feature_name})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_full_metric(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    full = summary[summary["scenario"] == "full_external"].copy()
    full["feature_set"] = pd.Categorical(full["feature_set"], categories=FEATURE_SET_ORDER, ordered=True)
    full = full.sort_values(["feature_set", "model"])

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x_positions = range(len(FEATURE_SET_ORDER))
    width = 0.36

    for offset, model_name in [(-width / 2, "baseline_full_train"), (width / 2, "robust_random_train")]:
        values = []
        for feature_set in FEATURE_SET_ORDER:
            row = full[(full["feature_set"] == feature_set) & (full["model"] == model_name)]
            values.append(float(row[metric].iloc[0]) if not row.empty else float("nan"))
        ax.bar([x + offset for x in x_positions], values, width=width, label=model_name)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(FEATURE_SET_ORDER, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Full ASAS-SN external {metric} by feature set")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scenario_metric(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    x_positions = list(range(len(SCENARIO_ORDER)))

    for (feature_set, model_name), sub in summary.groupby(["feature_set", "model"]):
        sub = sub.copy()
        sub["scenario"] = pd.Categorical(sub["scenario"], categories=SCENARIO_ORDER, ordered=True)
        sub = sub.sort_values("scenario")
        label = f"{feature_set}: {model_name}"
        ax.plot(x_positions, sub[metric], marker="o", linewidth=1.7, label=label)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(SCENARIO_ORDER, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"ASAS-SN external feature ablation {metric} by scenario")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    if not OGLE_TRAIN_PATH.exists():
        print("Missing OGLE training data. Run python scripts/03_download_ogle_lmc_sample.py")
        return
    if not EXTERNAL_TEST_PATH.exists():
        print("Missing ASAS-SN external test data. Run python scripts/07_download_asassn_external_test.py")
        return

    out_tab = PROJECT_ROOT / "outputs" / "tables"
    out_fig = PROJECT_ROOT / "outputs" / "figures"
    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading OGLE train and ASAS-SN external test light curves...")
    train_lc = load_real_lightcurves_from_csv(OGLE_TRAIN_PATH)
    external_lc = load_real_lightcurves_from_csv(EXTERNAL_TEST_PATH)
    if train_lc.empty:
        raise SystemExit("No usable rows remain after cleaning real_lightcurves.csv.")
    if external_lc.empty:
        raise SystemExit("No usable rows remain after cleaning external_test_lightcurves.csv.")

    print("[2/6] Extracting full and degraded training features...")
    train_full_features = extract_features(train_lc)

    robust_train_parts = []
    for n_points in DEGRADED_POINTS:
        robust_train_parts.append(
            degrade_dataset(
                train_lc,
                n_points=n_points,
                seed=RANDOM_STATE + n_points,
                mode="random",
                extra_noise_scale=0.03,
            )
        )
    robust_train_lc = pd.concat(robust_train_parts, ignore_index=True)
    train_robust_features = extract_features(robust_train_lc)

    all_features = numeric_feature_columns(train_full_features)
    feature_sets = define_feature_sets(all_features)
    save_feature_sets(feature_sets, out_tab / "external_feature_sets.csv")

    print("[3/6] Extracting external scenario features...")
    scenario_features = {
        "full_external": ("full", extract_features(external_lc)),
    }
    for n_points in DEGRADED_POINTS:
        scenario = f"early_{n_points}"
        degraded_external = degrade_dataset(
            external_lc,
            n_points=n_points,
            seed=RANDOM_STATE + 3000 + n_points,
            mode="early",
            extra_noise_scale=0.0,
        )
        scenario_features[scenario] = (n_points, extract_features(degraded_external))

    print("[4/6] Training and evaluating feature sets...")
    metric_rows = []

    for feature_set_name in FEATURE_SET_ORDER:
        feature_columns = feature_sets[feature_set_name]
        if not feature_columns:
            print(f"Warning: feature set {feature_set_name} has no columns; skipping.")
            continue

        X_train_full, y_train_full = prepare_xy(train_full_features, feature_columns)
        X_train_robust, y_train_robust = prepare_xy(train_robust_features, feature_columns)

        baseline = make_model(RANDOM_STATE)
        baseline.fit(X_train_full, y_train_full)

        robust = make_model(RANDOM_STATE + 1)
        robust.fit(X_train_robust, y_train_robust)

        models = [
            ("baseline_full_train", "ogle_full_train", baseline),
            ("robust_random_train", "ogle_random_degraded_train", robust),
        ]

        for scenario in SCENARIO_ORDER:
            n_points, features = scenario_features[scenario]
            X_test, y_test = prepare_xy(features, feature_columns)

            for model_name, train_regime, model in models:
                pred = model.predict(X_test)
                proba = model.predict_proba(X_test)
                row = compute_model_metrics(
                    model_name=model_name,
                    n_points=n_points,
                    y_true=y_test,
                    y_pred=pred,
                    proba=proba,
                    classes=model.classes_,
                )
                row["feature_set"] = feature_set_name
                row["train_regime"] = train_regime
                row["scenario"] = scenario
                row["n_features"] = len(feature_columns)
                metric_rows.append(row)

    print("[5/6] Saving tables...")
    summary = pd.DataFrame(metric_rows)
    summary = summary[
        [
            "feature_set",
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
            "n_features",
        ]
    ]
    summary["feature_set"] = pd.Categorical(summary["feature_set"], categories=FEATURE_SET_ORDER, ordered=True)
    summary["scenario"] = pd.Categorical(summary["scenario"], categories=SCENARIO_ORDER, ordered=True)
    summary = summary.sort_values(["feature_set", "scenario", "model"]).reset_index(drop=True)
    summary.to_csv(out_tab / "external_feature_ablation_summary.csv", index=False)

    delta = make_delta_table(summary)
    delta.to_csv(out_tab / "external_feature_ablation_delta.csv", index=False)

    print("[6/6] Saving plots...")
    plot_full_metric(summary, "accuracy", out_fig / "external_feature_ablation_full_accuracy.png")
    plot_full_metric(summary, "f1_macro", out_fig / "external_feature_ablation_full_f1.png")
    plot_scenario_metric(summary, "accuracy", out_fig / "external_feature_ablation_accuracy_by_scenario.png")
    plot_scenario_metric(summary, "f1_macro", out_fig / "external_feature_ablation_f1_by_scenario.png")

    print("\nDone.")
    print(f"Tables saved to: {out_tab}")
    print(f"Figures saved to: {out_fig}")
    print("\nOpen outputs/tables/external_feature_ablation_summary.csv first.")


if __name__ == "__main__":
    main()
