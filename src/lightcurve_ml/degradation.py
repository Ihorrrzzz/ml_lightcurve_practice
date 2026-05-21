from __future__ import annotations

import numpy as np
import pandas as pd


def degrade_lightcurve(
    lc: pd.DataFrame,
    n_points: int,
    seed: int | None = None,
    mode: str = "random",
    extra_noise_scale: float = 0.0,
) -> pd.DataFrame:
    """Return a degraded copy of one object's light curve.

    Parameters
    ----------
    lc:
        DataFrame for a single object.
    n_points:
        Maximum number of retained observations.
    seed:
        Random seed. Used by random point selection, contiguous block
        selection, and optional additional noise.
    mode:
        "random" keeps random points.
        "early" keeps the first n_points by time.
        "contiguous" keeps a random contiguous block after sorting by time.
    extra_noise_scale:
        Additional Gaussian noise added to magnitudes, in magnitudes.
    """
    if len(lc) == 0:
        return lc.copy()

    rng = np.random.default_rng(seed)
    lc_sorted = lc.sort_values("time").copy()

    keep_n = min(n_points, len(lc_sorted))
    if mode == "random":
        idx = rng.choice(lc_sorted.index.to_numpy(), size=keep_n, replace=False)
        out = lc_sorted.loc[idx].sort_values("time").copy()
    elif mode == "early":
        out = lc_sorted.head(keep_n).copy()
    elif mode == "contiguous":
        if len(lc_sorted) <= n_points:
            out = lc_sorted.copy()
        else:
            max_start = len(lc_sorted) - keep_n
            start = int(rng.integers(0, max_start + 1))
            out = lc_sorted.iloc[start : start + keep_n].copy()
    else:
        raise ValueError("mode must be 'random', 'early', or 'contiguous'")

    if extra_noise_scale > 0:
        out["mag"] = out["mag"] + rng.normal(0, extra_noise_scale, size=len(out))
        out["mag_err"] = np.sqrt(out["mag_err"].to_numpy() ** 2 + extra_noise_scale ** 2)

    return out.reset_index(drop=True)


def degrade_dataset(
    df: pd.DataFrame,
    n_points: int,
    seed: int = 42,
    mode: str = "random",
    extra_noise_scale: float = 0.0,
) -> pd.DataFrame:
    """Apply light-curve degradation independently to every object.

    Supported modes are "random", "early", and "contiguous"; see
    ``degrade_lightcurve`` for their meanings.
    """
    parts = []
    for i, (_, lc) in enumerate(df.groupby("object_id", sort=False)):
        parts.append(
            degrade_lightcurve(
                lc,
                n_points=n_points,
                seed=seed + i,
                mode=mode,
                extra_noise_scale=extra_noise_scale,
            )
        )
    return pd.concat(parts, ignore_index=True)
