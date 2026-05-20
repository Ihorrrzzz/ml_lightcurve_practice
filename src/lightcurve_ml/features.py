from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis


def _safe_float(value: float, fallback: float = np.nan) -> float:
    try:
        if np.isfinite(value):
            return float(value)
        return fallback
    except Exception:
        return fallback


def _lomb_scargle_features(time: np.ndarray, mag: np.ndarray, mag_err: np.ndarray) -> dict[str, float]:
    """Compute simple Lomb-Scargle period features.

    We use magnitudes centered around their mean. Lower magnitude = brighter,
    but period search is insensitive to sign for our purposes.
    """
    if len(time) < 8 or np.nanstd(mag) == 0:
        return {"ls_period": np.nan, "ls_power": np.nan}

    t_span = np.nanmax(time) - np.nanmin(time)
    if not np.isfinite(t_span) or t_span <= 0:
        return {"ls_period": np.nan, "ls_power": np.nan}

    min_period = 0.2
    max_period = max(0.25, min(60.0, 0.9 * t_span))
    if max_period <= min_period:
        return {"ls_period": np.nan, "ls_power": np.nan}

    min_frequency = 1.0 / max_period
    max_frequency = 1.0 / min_period

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ls = LombScargle(time, mag - np.nanmean(mag), mag_err)
            freq, power = ls.autopower(
                minimum_frequency=min_frequency,
                maximum_frequency=max_frequency,
                samples_per_peak=5,
                nyquist_factor=3,
            )
        if len(power) == 0:
            return {"ls_period": np.nan, "ls_power": np.nan}
        best = int(np.nanargmax(power))
        return {"ls_period": float(1.0 / freq[best]), "ls_power": float(power[best])}
    except Exception:
        return {"ls_period": np.nan, "ls_power": np.nan}


def extract_features_one(lc: pd.DataFrame) -> dict[str, float | str]:
    """Extract interpretable features from one light curve."""
    object_id = str(lc["object_id"].iloc[0])
    label = str(lc["label"].iloc[0]) if "label" in lc.columns else ""

    time = lc["time"].to_numpy(dtype=float)
    mag = lc["mag"].to_numpy(dtype=float)
    err = lc["mag_err"].to_numpy(dtype=float) if "mag_err" in lc.columns else np.full(len(lc), np.nan)

    order = np.argsort(time)
    time, mag, err = time[order], mag[order], err[order]

    n = len(mag)
    t_span = np.nanmax(time) - np.nanmin(time) if n else np.nan
    amp = np.nanpercentile(mag, 95) - np.nanpercentile(mag, 5) if n >= 3 else np.nan
    median_err = np.nanmedian(err) if n else np.nan

    if n >= 2:
        dt = np.diff(time)
        dm = np.diff(mag)
        good = dt > 0
        slopes = dm[good] / dt[good] if np.any(good) else np.array([np.nan])
        max_abs_slope = np.nanmax(np.abs(slopes))
        median_cadence = np.nanmedian(dt[good]) if np.any(good) else np.nan
    else:
        max_abs_slope = np.nan
        median_cadence = np.nan

    feats = {
        "object_id": object_id,
        "label": label,
        "n_points": float(n),
        "time_span": _safe_float(t_span),
        "mean_mag": _safe_float(np.nanmean(mag)),
        "median_mag": _safe_float(np.nanmedian(mag)),
        "std_mag": _safe_float(np.nanstd(mag)),
        "amplitude_p95_p5": _safe_float(amp),
        "median_err": _safe_float(median_err),
        "amp_to_err": _safe_float(amp / median_err if median_err and median_err > 0 else np.nan),
        "skew_mag": _safe_float(skew(mag, nan_policy="omit") if n >= 4 else np.nan),
        "kurtosis_mag": _safe_float(kurtosis(mag, nan_policy="omit") if n >= 4 else np.nan),
        "max_abs_slope": _safe_float(max_abs_slope),
        "median_cadence": _safe_float(median_cadence),
    }
    feats.update(_lomb_scargle_features(time, mag, err))
    return feats


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for every object in a light-curve table."""
    rows = [extract_features_one(lc) for _, lc in df.groupby("object_id", sort=False)]
    features = pd.DataFrame(rows)

    # Replace infinities, then median-impute numeric columns.
    features = features.replace([np.inf, -np.inf], np.nan)
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if features[col].isna().all():
            features[col] = 0.0
        else:
            features[col] = features[col].fillna(features[col].median())

    return features
