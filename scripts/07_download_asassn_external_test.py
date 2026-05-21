from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Allow running from project root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightcurve_ml.asassn import build_asassn_external_test_set


SELECTED_PATH = PROJECT_ROOT / "data" / "processed" / "asassn_external_selected_objects.csv"


def main() -> None:
    external = build_asassn_external_test_set(n_per_class=30)

    print("\nASAS-SN external test set summary")
    print(f"Rows: {len(external)}")
    print(f"Objects: {external['object_id'].nunique() if not external.empty else 0}")

    print("\nObjects by label:")
    if external.empty:
        print("  none")
    else:
        object_counts = external[["label", "object_id"]].drop_duplicates().groupby("label").size().sort_index()
        for label, count in object_counts.items():
            print(f"  {label}: {count}")

    print("\nRows by label:")
    if external.empty:
        print("  none")
    else:
        for label, count in external.groupby("label").size().sort_index().items():
            print(f"  {label}: {count}")

    print("\nBand/camera counts:")
    if external.empty:
        print("  none")
    else:
        for band, count in external["band"].value_counts().sort_index().items():
            print(f"  {band}: {count}")

    print("\nPoints per object by label:")
    if external.empty:
        print("  none")
    else:
        points = external.groupby(["label", "object_id"]).size().reset_index(name="n_points")
        summary = points.groupby("label")["n_points"].agg(["median", "min", "max"]).sort_index()
        for label, row in summary.iterrows():
            print(
                f"  {label}: median={row['median']:.1f}, "
                f"min={int(row['min'])}, max={int(row['max'])}"
            )

    if SELECTED_PATH.exists():
        selected = pd.read_csv(SELECTED_PATH)
        failed = selected[selected["download_status"] != "ok"]
        print("\nDownload status:")
        print(f"  selected objects: {len(selected)}")
        print(f"  successful objects: {int((selected['download_status'] == 'ok').sum())}")
        print(f"  failed or empty objects: {len(failed)}")
        if not failed.empty:
            for _, row in failed.iterrows():
                print(f"  {row['target_label']} {row['asassn_name']} ({row['source_id']}): {row['download_status']}")

    print("\nNext run: python scripts/05_run_external_validation.py")


if __name__ == "__main__":
    main()
