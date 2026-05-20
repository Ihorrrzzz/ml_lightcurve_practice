from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_metric_vs_npoints(summary, metric: str, out_path: str | Path) -> None:
    """Plot one metric as a function of retained points."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model_name, sub in summary.groupby("model"):
        sub = sub.sort_values("n_points")
        ax.plot(sub["n_points"], sub[metric], marker="o", label=model_name)

    ax.set_xlabel("Кількість точок у деградованій кривій")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} залежно від кількості точок")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, title: str, out_path: str | Path) -> None:
    """Save confusion matrix plot."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=35, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
