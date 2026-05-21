from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from lightcurve_ml.plotting import plot_confusion_matrix


def safe_log_loss(y_true, proba, labels) -> float:
    """Compute log loss, returning NaN when a split lacks required classes."""
    try:
        return float(log_loss(y_true, proba, labels=list(labels)))
    except Exception:
        return float("nan")


def multiclass_brier_score(y_true, proba, labels) -> float:
    """Return the multiclass Brier score using one-hot class targets."""
    labels = list(labels)
    proba = np.asarray(proba, dtype=float)

    if proba.shape != (len(y_true), len(labels)):
        return float("nan")

    y_series = pd.Series(y_true)
    if set(y_series.unique()) - set(labels):
        return float("nan")

    truth = pd.get_dummies(y_series).reindex(columns=labels, fill_value=0)
    truth = truth.to_numpy(dtype=float)
    return float(np.mean(np.sum((proba - truth) ** 2, axis=1)))


def compute_model_metrics(
    model_name: str,
    n_points: int | str,
    y_true,
    y_pred,
    proba,
    classes,
    confidence_threshold: float = 0.60,
) -> dict[str, float | int | str]:
    """Compute summary metrics for one model on one degraded test set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba, dtype=float)

    max_confidence = np.max(proba, axis=1)
    confident_mask = max_confidence >= confidence_threshold

    if np.any(confident_mask):
        confident_accuracy = accuracy_score(y_true[confident_mask], y_pred[confident_mask])
    else:
        confident_accuracy = float("nan")

    return {
        "model": model_name,
        "n_points": n_points,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": safe_log_loss(y_true, proba, labels=classes),
        "brier_score": multiclass_brier_score(y_true, proba, labels=classes),
        "confidence_threshold": float(confidence_threshold),
        "abstention_rate": 1.0 - float(np.mean(confident_mask)),
        "confident_accuracy": float(confident_accuracy),
        "n_test_objects": int(len(y_true)),
    }


def per_class_report_table(
    dataset_name: str,
    model_name: str,
    n_points: int | str,
    y_true,
    y_pred,
    labels,
) -> pd.DataFrame:
    """Return one row per class with precision, recall, F1, and support."""
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    rows = []
    for label, p_value, r_value, f_value, s_value in zip(labels, precision, recall, f1, support):
        rows.append(
            {
                "dataset_name": dataset_name,
                "model": model_name,
                "n_points": n_points,
                "label": label,
                "precision": float(p_value),
                "recall": float(r_value),
                "f1": float(f_value),
                "support": int(s_value),
            }
        )
    return pd.DataFrame(rows)


def dataset_summary_table(lightcurves: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Summarize row/object counts and per-label sampling density."""
    labels = sorted(lightcurves["label"].dropna().unique())
    n_rows = int(len(lightcurves))
    n_objects = int(lightcurves["object_id"].nunique())
    n_classes = int(len(labels))

    rows = []
    for label in labels:
        label_curves = lightcurves[lightcurves["label"] == label]
        points_per_object = label_curves.groupby("object_id").size()

        rows.append(
            {
                "dataset_name": dataset_name,
                "n_rows": n_rows,
                "n_objects": n_objects,
                "n_classes": n_classes,
                "label": label,
                "n_objects_label": int(points_per_object.size),
                "n_rows_label": int(len(label_curves)),
                "median_points_per_object_label": float(points_per_object.median()),
                "min_points_per_object_label": int(points_per_object.min()),
                "max_points_per_object_label": int(points_per_object.max()),
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    title: str,
    out_path: str | Path,
) -> np.ndarray:
    """Compute and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plot_confusion_matrix(cm, labels, title=title, out_path=out_path)
    return cm


def compute_reliability_curve(
    y_true,
    y_pred,
    proba,
    bins: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute empirical accuracy by confidence bin."""
    if bins is None:
        bins = np.linspace(0.0, 1.0, 11)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba, dtype=float)

    confidence = np.max(proba, axis=1)
    correct = (y_true == y_pred).astype(float)

    rows = []
    for i in range(len(bins) - 1):
        left = float(bins[i])
        right = float(bins[i + 1])
        if i == len(bins) - 2:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        n_bin = int(np.sum(mask))
        if n_bin:
            mean_confidence = float(np.mean(confidence[mask]))
            empirical_accuracy = float(np.mean(correct[mask]))
        else:
            mean_confidence = float("nan")
            empirical_accuracy = float("nan")

        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "bin_center": (left + right) / 2.0,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "n_objects": n_bin,
            }
        )

    return pd.DataFrame(rows)


def plot_reliability_curves(
    curves: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str = "Reliability curve",
) -> None:
    """Plot reliability curves for one or more models."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.4", label="perfect calibration")

    for model_name, curve in curves.items():
        valid = curve["n_objects"] > 0
        sub = curve.loc[valid]
        ax.plot(
            sub["mean_confidence"],
            sub["empirical_accuracy"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_latex_summary(summary: pd.DataFrame, out_path: str | Path) -> None:
    """Save the compact report-ready metric table as LaTeX."""
    columns = [
        "model",
        "n_points",
        "accuracy",
        "f1_macro",
        "log_loss",
        "brier_score",
        "abstention_rate",
        "confident_accuracy",
    ]
    save_latex_table(summary, out_path, columns=columns)


def save_latex_table(
    table: pd.DataFrame,
    out_path: str | Path,
    columns: list[str] | None = None,
) -> None:
    """Save a DataFrame as LaTeX using pandas, with a small local fallback."""
    if columns is not None:
        table = table.loc[:, columns]

    try:
        latex_table = table.to_latex(index=False, float_format="%.3f")
    except ImportError:
        latex_table = _simple_latex_table(table)
    Path(out_path).write_text(latex_table, encoding="utf-8")


def _simple_latex_table(table: pd.DataFrame) -> str:
    """Small fallback for environments where pandas lacks its LaTeX backend."""
    column_spec = "l" * len(table.columns)
    lines = [
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(_escape_latex(str(col)) for col in table.columns) + r" \\",
        "\\midrule",
    ]

    for _, row in table.iterrows():
        values = [_format_latex_value(row[col]) for col in table.columns]
        lines.append(" & ".join(values) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _format_latex_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return _escape_latex(str(value))


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)
