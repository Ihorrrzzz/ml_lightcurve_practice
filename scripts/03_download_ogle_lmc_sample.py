from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightcurve_ml.ogle import build_ogle_lmc_sample


def main() -> None:
    # Small reproducible OGLE-IV LMC I-band MVP sample, not the full catalog.
    sample = build_ogle_lmc_sample(n_per_class=30)

    object_counts = (
        sample[["label", "object_id"]]
        .drop_duplicates()
        .groupby("label")
        .size()
        .sort_index()
    )

    print("Saved data/processed/real_lightcurves.csv")
    print(f"Rows: {len(sample)}")
    print(f"Objects: {sample['object_id'].nunique()}")
    print("Objects by label:")
    for label, count in object_counts.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
