from __future__ import annotations

import numpy as np
import pandas as pd


CLASSES = ["RR_Lyrae", "Cepheid", "Eclipsing", "Transient", "AGN"]


def _rr_lyrae(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    period = rng.uniform(0.35, 0.85)
    phase = (t / period + rng.uniform(0, 1)) % 1.0
    # Asymmetric sawtooth-like pulsation: fast rise, slower decline
    shape = np.where(phase < 0.18, -1.0 + phase / 0.18 * 2.0, 1.0 - (phase - 0.18) / 0.82 * 2.0)
    return 15.0 + rng.normal(0, 0.25) + rng.uniform(0.25, 0.55) * shape


def _cepheid(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    period = rng.uniform(2.0, 18.0)
    phase = 2.0 * np.pi * (t / period + rng.uniform(0, 1))
    return 14.5 + rng.normal(0, 0.25) + rng.uniform(0.25, 0.75) * (
        np.sin(phase) + 0.25 * np.sin(2 * phase + 0.7)
    )


def _eclipsing(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    period = rng.uniform(0.6, 8.0)
    phase = (t / period + rng.uniform(0, 1)) % 1.0
    base = 14.8 + rng.normal(0, 0.25)
    primary = rng.uniform(0.5, 1.2) * np.exp(-0.5 * ((phase - 0.0) / 0.035) ** 2)
    primary += rng.uniform(0.5, 1.2) * np.exp(-0.5 * ((phase - 1.0) / 0.035) ** 2)
    secondary = rng.uniform(0.15, 0.55) * np.exp(-0.5 * ((phase - 0.5) / 0.045) ** 2)
    # Magnitudes increase during eclipse
    return base + primary + secondary


def _transient(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t0 = rng.uniform(0.1, 0.5) * (t.max() - t.min()) + t.min()
    rise = rng.uniform(1.5, 8.0)
    decay = rng.uniform(15.0, 45.0)
    amp = rng.uniform(1.0, 2.8)
    dt = t - t0
    flux_like = np.where(dt < 0, np.exp(dt / rise), np.exp(-dt / decay))
    # Brighter means lower magnitude
    return 16.2 + rng.normal(0, 0.3) - amp * flux_like


def _agn(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Smooth random-walk-like variability
    steps = rng.normal(0, rng.uniform(0.03, 0.08), size=len(t))
    walk = np.cumsum(steps)
    walk = walk - np.mean(walk)
    return 15.5 + rng.normal(0, 0.4) + walk


def _generate_signal(label: str, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if label == "RR_Lyrae":
        return _rr_lyrae(t, rng)
    if label == "Cepheid":
        return _cepheid(t, rng)
    if label == "Eclipsing":
        return _eclipsing(t, rng)
    if label == "Transient":
        return _transient(t, rng)
    if label == "AGN":
        return _agn(t, rng)
    raise ValueError(f"Unknown class: {label}")


def generate_synthetic_lightcurves(
    n_per_class: int = 80,
    n_points: int = 120,
    seed: int = 42,
    t_max: float = 120.0,
) -> pd.DataFrame:
    """Generate a synthetic labelled light-curve table.

    Columns:
        object_id, label, time, mag, mag_err, band

    The curves are not intended to be physically precise. They only imitate
    broad morphology differences so that the MVP pipeline can be tested.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    object_counter = 0

    for label in CLASSES:
        for _ in range(n_per_class):
            object_counter += 1
            object_id = f"{label}_{object_counter:04d}"

            # Uneven cadence with random missing intervals
            t = np.sort(rng.uniform(0, t_max, size=n_points))
            mask = rng.random(n_points) > rng.uniform(0.0, 0.12)
            t = t[mask]
            if len(t) < 30:
                t = np.sort(rng.uniform(0, t_max, size=30))

            true_mag = _generate_signal(label, t, rng)

            # Heteroscedastic photometric uncertainty
            mag_err = rng.uniform(0.015, 0.08, size=len(t))
            mag = true_mag + rng.normal(0, mag_err)

            for ti, mi, ei in zip(t, mag, mag_err):
                records.append(
                    {
                        "object_id": object_id,
                        "label": label,
                        "time": float(ti),
                        "mag": float(mi),
                        "mag_err": float(ei),
                        "band": "V",
                    }
                )

    return pd.DataFrame.from_records(records)
