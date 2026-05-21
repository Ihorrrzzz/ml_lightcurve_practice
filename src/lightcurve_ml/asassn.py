from __future__ import annotations

from io import StringIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "asassn_v"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "external_test_lightcurves.csv"
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "data" / "processed" / "asassn_external_selected_objects.csv"

ASASSN_CLASS_CONFIG = {
    "RRAB": {
        "catalog_path": RAW_DIR / "asassn_rrab_catalog.csv",
        "variable_type": "RRAB",
        "target_label": "rrlyr",
    },
    "DCEP": {
        "catalog_path": RAW_DIR / "asassn_dcep_catalog.csv",
        "variable_type": "DCEP",
        "target_label": "cep",
    },
    "EW": {
        "catalog_path": RAW_DIR / "asassn_ew_catalog.csv",
        "variable_type": "EW",
        "target_label": "ecl",
    },
}

LIGHTCURVE_COLUMNS = ["object_id", "label", "time", "mag", "mag_err", "band"]
CATALOG_REQUIRED_COLUMNS = ["source_id", "asassn_name", "raj2000", "dej2000"]
ASASSN_LIGHTCURVE_COLUMNS = ["hjd", "camera", "mag", "mag_err"]
REQUEST_HEADERS = {"User-Agent": "ml-lightcurve-practice/0.1 ASAS-SN external dataset builder"}


def select_asassn_objects(
    catalog_path,
    variable_type,
    n_objects: int = 30,
    min_class_probability: float = 0.95,
) -> pd.DataFrame:
    """Select high-confidence ASAS-SN objects from one metadata catalog."""
    path = Path(catalog_path)
    catalog = pd.read_csv(path, dtype={"source_id": "string", "asassn_name": "string"})

    missing = [col for col in [*CATALOG_REQUIRED_COLUMNS, "variable_type", "class_probability"] if col not in catalog]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

    clean = catalog[catalog["variable_type"] == variable_type].copy()
    clean["class_probability"] = pd.to_numeric(clean["class_probability"], errors="coerce")
    clean = clean.dropna(subset=CATALOG_REQUIRED_COLUMNS).copy()

    high_conf = clean[clean["class_probability"] >= min_class_probability].copy()
    if len(high_conf) < n_objects:
        print(
            f"Warning: {path.name} has only {len(high_conf)} {variable_type} objects with "
            f"class_probability >= {min_class_probability}. Taking best available rows."
        )
        selected_pool = clean
    else:
        selected_pool = high_conf

    selected = selected_pool.sort_values(
        ["class_probability", "asassn_name"],
        ascending=[False, True],
        na_position="last",
    ).head(n_objects)

    return selected.reset_index(drop=True)


def download_asassn_lightcurve(source_id, asassn_name, label) -> pd.DataFrame:
    """Download and normalize one ASAS-SN light curve."""
    source_id = _clean_text_id(source_id)
    object_id = _clean_text_id(asassn_name) or source_id
    url = f"https://asas-sn.osu.edu/variables/{source_id}.csv"

    last_error = ""
    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            if response.status_code == 200 and response.text.strip():
                return _parse_asassn_lightcurve_csv(response.text, object_id=object_id, label=label)
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)

        if attempt < 3:
            time.sleep(0.5 * attempt)

    print(f"Warning: failed to download {object_id} from {url}: {last_error}")
    return pd.DataFrame(columns=LIGHTCURVE_COLUMNS)


def build_asassn_external_test_set(
    n_per_class: int = 30,
    output_path="data/processed/external_test_lightcurves.csv",
) -> pd.DataFrame:
    """Build and save a normalized ASAS-SN external validation dataset."""
    output_path = _resolve_project_path(output_path)
    selected_path = output_path.parent / "asassn_external_selected_objects.csv"

    selected_parts = []
    curve_parts = []
    metadata_rows = []

    for _, config in ASASSN_CLASS_CONFIG.items():
        selected = select_asassn_objects(
            catalog_path=config["catalog_path"],
            variable_type=config["variable_type"],
            n_objects=n_per_class,
        )
        selected = selected.copy()
        selected["target_label"] = config["target_label"]
        selected_parts.append(selected)

        print(f"Downloading {len(selected)} {config['variable_type']} -> {config['target_label']} light curves...")
        for _, row in selected.iterrows():
            source_id = _clean_text_id(row["source_id"])
            asassn_name = _clean_text_id(row["asassn_name"])
            curve = download_asassn_lightcurve(
                source_id=source_id,
                asassn_name=asassn_name,
                label=config["target_label"],
            )
            curve_parts.append(curve)
            metadata_rows.append(
                {
                    "source_id": source_id,
                    "asassn_name": asassn_name,
                    "variable_type": config["variable_type"],
                    "target_label": config["target_label"],
                    "raj2000": row["raj2000"],
                    "dej2000": row["dej2000"],
                    "class_probability": row["class_probability"],
                    "download_status": "ok" if not curve.empty else "failed_or_empty",
                    "n_points": int(len(curve)),
                }
            )
            time.sleep(0.2)

    if curve_parts:
        external = pd.concat(curve_parts, ignore_index=True)
    else:
        external = pd.DataFrame(columns=LIGHTCURVE_COLUMNS)

    selected_metadata = pd.DataFrame(metadata_rows)
    if selected_parts and selected_metadata.empty:
        selected_metadata = pd.concat(selected_parts, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    external.to_csv(output_path, index=False)
    selected_metadata.to_csv(selected_path, index=False)

    print(f"Saved normalized ASAS-SN external test set to {output_path}")
    print(f"Saved selected-object metadata to {selected_path}")
    return external


def _parse_asassn_lightcurve_csv(csv_text: str, object_id: str, label: str) -> pd.DataFrame:
    raw = pd.read_csv(StringIO(csv_text))
    missing = [col for col in ASASSN_LIGHTCURVE_COLUMNS if col not in raw.columns]
    if missing:
        print(f"Warning: {object_id} light curve missing column(s): {', '.join(missing)}")
        return pd.DataFrame(columns=LIGHTCURVE_COLUMNS)

    clean = raw.loc[:, ASASSN_LIGHTCURVE_COLUMNS].copy()
    for col in ("hjd", "mag", "mag_err"):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=["hjd", "mag", "mag_err", "camera"]).copy()
    finite_mask = (
        np.isfinite(clean["hjd"])
        & np.isfinite(clean["mag"])
        & np.isfinite(clean["mag_err"])
        & (clean["mag_err"] > 0)
        & (clean["mag"] > -5)
        & (clean["mag"] < 30)
    )
    clean = clean.loc[finite_mask].copy()

    if clean.empty:
        return pd.DataFrame(columns=LIGHTCURVE_COLUMNS)

    out = pd.DataFrame(
        {
            "object_id": object_id,
            "label": label,
            "time": clean["hjd"].astype(float),
            "mag": clean["mag"].astype(float),
            "mag_err": clean["mag_err"].astype(float),
            "band": clean["camera"].astype(str),
        }
    )
    return out.loc[:, LIGHTCURVE_COLUMNS].sort_values("time").reset_index(drop=True)


def _resolve_project_path(path_like) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _clean_text_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text
