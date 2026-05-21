from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("object_id", "label", "time", "mag", "mag_err", "band")


def load_real_lightcurves_from_csv(path: str | Path) -> pd.DataFrame:
    """Load normalized real light curves from a CSV file.

    The input CSV must contain one observation per row with these columns:
    ``object_id``, ``label``, ``time``, ``mag``, ``mag_err``, and ``band``.
    Rows missing ``object_id``, ``label``, ``time``, or ``mag`` are removed.
    ``time``, ``mag``, and ``mag_err`` are converted to numeric values.
    Missing or non-numeric ``mag_err`` values are filled with the median
    available uncertainty, or ``0.05`` if no usable uncertainty exists.

    Parameters
    ----------
    path:
        Path to the normalized real-light-curve CSV.

    Returns
    -------
    pandas.DataFrame
        Clean dataframe sorted by ``object_id`` and ``time`` with exactly the
        six required columns.
    """
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        expected = ", ".join(REQUIRED_COLUMNS)
        raise ValueError(f"Missing required column(s): {missing}. Expected columns: {expected}.")

    clean = df.loc[:, REQUIRED_COLUMNS].copy()

    for col in ("time", "mag", "mag_err"):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=["object_id", "label", "time", "mag"]).copy()

    median_mag_err = clean["mag_err"].median(skipna=True)
    if not np.isfinite(median_mag_err):
        median_mag_err = 0.05
    clean["mag_err"] = clean["mag_err"].fillna(float(median_mag_err))

    clean = clean.sort_values(["object_id", "time"]).reset_index(drop=True)
    return clean.loc[:, REQUIRED_COLUMNS]
