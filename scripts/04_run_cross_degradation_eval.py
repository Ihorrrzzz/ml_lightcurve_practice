from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Allow running from project root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightcurve_ml.degradation import degrade_dataset
from lightcurve_ml.evaluation import compute_model_metrics, save_latex_table
from lightcurve_ml.features import extract_features
from lightcurve_ml.real_data import load_real_lightcurves_from_csv


RANDOM_STATE = 42
DEGRADED_POINTS = [5, 10, 20, 50]
REAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "real_lightcurves.csv"
TEST_SCENARIOS = [
    {"test_mode": "random", "mode": "random", "extra_noise_scale": 0.03},
    {"test_mode": "early", "mode": "early", "extra_noise_scale": 0.03},
    {"test_mode": "contiguous", "mode": "contiguous", "extra_noise_scale": 0.03},
    {"test_mode": "random_high_noise", "mode": "random", "extra_noise_scale": 0.08},
    {"test_mode": "early_high_noise", "mode": "early", "extra_noise_scale": 0.08},
]


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
    X = features.drop(columns=["object_id", "label"])
    return X, y


def plot_cross_metric(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Plot one metric for every model and test-mode combination."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for (model_name, test_mode), sub in summary.groupby(["model", "test_mode"]):
        sub = sub.sort_values("n_points")
        label = f"{model_name}: {test_mode}"
        ax.plot(sub["n_points"], sub[metric], marker="o", linewidth=1.8, label=label)

    ax.set_xlabel("Number of retained points")
    ax.set_ylabel(metric)
    ax.set_title(f"Real cross-degradation {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cross_metric_by_mode(summary: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """Save one compact plot per test mode."""
    metric_name = "f1" if metric == "f1_macro" else metric

    for test_mode, sub_mode in summary.groupby("test_mode"):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))

        for model_name, sub in sub_mode.groupby("model"):
            sub = sub.sort_values("n_points")
            ax.plot(sub["n_points"], sub[metric], marker="o", linewidth=2.0, label=model_name)

        ax.set_xlabel("Number of retained points")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: {test_mode}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"real_cross_{metric_name}_{test_mode}.png", dpi=180)
        plt.close(fig)


def make_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare robust_random_train against baseline_full_train for each scenario."""
    rows = []
    for (test_mode, n_points), sub in summary.groupby(["test_mode", "n_points"]):
        by_model = sub.set_index("model")
        baseline = by_model.loc["baseline_full_train"]
        robust = by_model.loc["robust_random_train"]
        rows.append(
            {
                "test_mode": test_mode,
                "n_points": int(n_points),
                "baseline_accuracy": float(baseline["accuracy"]),
                "robust_accuracy": float(robust["accuracy"]),
                "accuracy_delta": float(robust["accuracy"] - baseline["accuracy"]),
                "baseline_f1_macro": float(baseline["f1_macro"]),
                "robust_f1_macro": float(robust["f1_macro"]),
                "f1_delta": float(robust["f1_macro"] - baseline["f1_macro"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_mode", "n_points"]).reset_index(drop=True)


def main() -> None:
    if not REAL_DATA_PATH.exists():
        print(
            f"Missing real-data CSV: {REAL_DATA_PATH}\n"
            "Run this command first:\n"
            "python scripts/03_download_ogle_lmc_sample.py"
        )
        return

    out_fig = PROJECT_ROOT / "outputs" / "figures"
    out_tab = PROJECT_ROOT / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading real light curves...")
    lightcurves = load_real_lightcurves_from_csv(REAL_DATA_PATH)
    if lightcurves.empty:
        raise SystemExit("No usable rows remain after cleaning real_lightcurves.csv.")

    print("[2/6] Splitting by object...")
    try:
        train_lc, test_lc = split_by_object(lightcurves)
    except ValueError as exc:
        raise SystemExit(
            "Could not create a stratified object-level train/test split. "
            "Check that each label has enough distinct object_id values."
        ) from exc

    print("[3/6] Training baseline model on full training curves...")
    train_full_features = extract_features(train_lc)
    X_train_full, y_train_full = prepare_xy(train_full_features)

    baseline = make_model(RANDOM_STATE)
    baseline.fit(X_train_full, y_train_full)

    print("[4/6] Training robust model on random degraded training curves...")
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
    robust_train_features = extract_features(robust_train_lc)
    X_train_robust, y_train_robust = prepare_xy(robust_train_features)

    robust = make_model(RANDOM_STATE + 1)
    robust.fit(X_train_robust, y_train_robust)

    print("[5/6] Evaluating cross-degradation scenarios...")
    models = [
        ("baseline_full_train", "full_train", baseline),
        ("robust_random_train", "random_degraded_train", robust),
    ]
    metric_rows = []

    for scenario_i, scenario in enumerate(TEST_SCENARIOS):
        for n_points in DEGRADED_POINTS:
            degraded_test = degrade_dataset(
                test_lc,
                n_points=n_points,
                seed=RANDOM_STATE + 1000 + scenario_i * 100 + n_points,
                mode=scenario["mode"],
                extra_noise_scale=scenario["extra_noise_scale"],
            )
            test_features = extract_features(degraded_test)
            X_test, y_test = prepare_xy(test_features)

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
                metrics["test_mode"] = scenario["test_mode"]
                metric_rows.append(metrics)

    summary = pd.DataFrame(metric_rows)
    summary = summary[
        [
            "model",
            "train_regime",
            "test_mode",
            "n_points",
            "accuracy",
            "f1_macro",
            "log_loss",
            "brier_score",
            "abstention_rate",
            "confident_accuracy",
            "n_test_objects",
        ]
    ].sort_values(["test_mode", "n_points", "model"])

    summary_path = out_tab / "real_cross_degradation_summary.csv"
    summary.to_csv(summary_path, index=False)
    save_latex_table(summary, out_tab / "real_cross_degradation_summary.tex")

    delta = make_delta_table(summary)
    delta.to_csv(out_tab / "real_cross_degradation_delta.csv", index=False)

    print("[6/6] Saving plots...")
    plot_cross_metric(summary, "accuracy", out_fig / "real_cross_accuracy_by_mode.png")
    plot_cross_metric(summary, "f1_macro", out_fig / "real_cross_f1_by_mode.png")
    plot_cross_metric_by_mode(summary, "accuracy", out_fig)
    plot_cross_metric_by_mode(summary, "f1_macro", out_fig)

    print("\nDone.")
    print(f"Tables saved to: {out_tab}")
    print(f"Figures saved to: {out_fig}")
    print(f"\nOpen {summary_path.relative_to(PROJECT_ROOT)} first.")


if __name__ == "__main__":
    main()
