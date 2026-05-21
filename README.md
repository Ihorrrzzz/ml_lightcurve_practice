# ML Light Curve Practice

Starter project for the scientific-practice MVP:

**Goal:** test how light-curve classification quality changes when curves become incomplete/sparse/noisy.

The first MVP uses synthetic light curves as a smoke test for the code structure:
- generate synthetic light curves for several astrophysical-like classes;
- extract interpretable features;
- train a baseline model on full curves;
- train a robust model on degraded curves;
- compare both models on degraded test sets;
- save plots and metric tables.

The next-stage scaffold can run the same experiment on normalized real light curves.

## Project structure

```text
ml_lightcurve_practice/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── 01_run_synthetic_mvp.py
│   ├── 02_run_real_data_mvp.py
│   ├── 03_download_ogle_lmc_sample.py
│   ├── 04_run_cross_degradation_eval.py
│   ├── 05_run_external_validation.py
│   ├── 06_investigate_asassn_access.py
│   └── 07_download_asassn_external_test.py
├── src/
│   └── lightcurve_ml/
│       ├── __init__.py
│       ├── asassn.py
│       ├── degradation.py
│       ├── evaluation.py
│       ├── features.py
│       ├── ogle.py
│       ├── plotting.py
│       ├── real_data.py
│       └── synthetic.py
├── requirements.txt
└── README.md
```

## Quick start

From the project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run MVP:

```bash
python scripts/01_run_synthetic_mvp.py
```

Expected outputs:
- `outputs/tables/synthetic_metrics.csv`
- `outputs/tables/synthetic_summary.csv`
- `outputs/figures/accuracy_vs_npoints.png`
- `outputs/figures/f1_vs_npoints.png`
- `outputs/figures/confusion_baseline_n10.png`
- `outputs/figures/confusion_robust_n10.png`

## Real data MVP

The real-data runner expects a normalized CSV at:

```text
data/processed/real_lightcurves.csv
```

The CSV must contain one observation per row with these columns:

```text
object_id,label,time,mag,mag_err,band
```

Column meaning:
- `object_id`: stable source identifier shared by all observations of one object;
- `label`: class label for that object;
- `time`: observation time;
- `mag`: measured magnitude;
- `mag_err`: magnitude uncertainty;
- `band`: photometric band or filter.

The loader validates the required columns, removes rows missing `object_id`, `label`, `time`, or `mag`, converts `time`, `mag`, and `mag_err` to numeric values, fills missing `mag_err` with the median available uncertainty or `0.05`, and sorts by `object_id` and `time`.

A header-only template is available at:

```text
data/processed/real_lightcurves_template.csv
```

A header-only template for an independent external test set is available at:

```text
data/processed/external_test_lightcurves_template.csv
```

Run the real-data MVP from the project root:

```bash
python scripts/02_run_real_data_mvp.py
```

Expected outputs:
- `outputs/tables/real_metrics.csv`
- `outputs/tables/real_summary.csv`
- `outputs/figures/real_accuracy_vs_npoints.png`
- `outputs/figures/real_f1_vs_npoints.png`

ASAS-SN, OGLE, Gaia-derived data, or other labelled public light-curve samples can later be converted into this normalized format.

## OGLE LMC sample

The project includes a small reproducible OGLE-IV OCVS LMC downloader for an I-band MVP sample. It downloads only the first 30 light curves for each selected class (`rrlyr`, `cep`, `ecl`) and writes the normalized CSV used by the real-data MVP. This is not the full OGLE catalog.

Install dependencies, download the sample, then run the real-data experiment:

```bash
pip install -r requirements.txt
python scripts/03_download_ogle_lmc_sample.py
python scripts/02_run_real_data_mvp.py
```

## Cross-degradation evaluation

The matched degradation experiment is useful because it controls the training and test conditions, but it can be optimistic: the robust model is trained on synthetic random degradation and then tested on very similar degradation.

Cross-degradation evaluation keeps the same object-level train/test split principle, trains the robust model only on random degraded training curves, and then tests both models under random, early, contiguous-window, and higher-noise degradation modes. This checks whether robust training helps under sparse/noisy light curves that do not exactly match the training degradation procedure.

Run it after creating `data/processed/real_lightcurves.csv`:

```bash
python scripts/04_run_cross_degradation_eval.py
```

## ASAS-SN external test set

The ASAS-SN external test set is built from local ASAS-SN Variable Stars metadata catalogs placed under:

```text
data/raw/asassn_v/
```

Expected catalog files:

```text
asassn_rrab_catalog.csv
asassn_dcep_catalog.csv
asassn_ew_catalog.csv
```

The downloader selects high-confidence objects, downloads light curves from ASAS-SN, and writes `data/processed/external_test_lightcurves.csv` in the normalized schema. Labels are mapped as `RRAB -> rrlyr`, `DCEP -> cep`, and `EW -> ecl`. This dataset is used only as an independent external validation set and is not used for training.

Run:

```bash
python scripts/07_download_asassn_external_test.py
python scripts/05_run_external_validation.py
```

## External validation

Controlled degradation is useful for isolating the effect of sparse and noisy observations, but it is not enough to demonstrate external generalization. A stronger validation trains on the OGLE sample and evaluates on an independent labelled dataset prepared in the same normalized format.

Create the external test CSV at:

```text
data/processed/external_test_lightcurves.csv
```

It must use the standard columns:

```text
object_id,label,time,mag,mag_err,band
```

The external data can come from ASAS-SN, Gaia, BHTOM, or another independent labelled source. The script does not split the external data and does not train on it.

Run:

```bash
python scripts/05_run_external_validation.py
```

## Interpretation

This synthetic MVP is not a final scientific result. It checks the experimental logic:
baseline trained on full data vs robust model trained on degraded data.

The real-data MVP keeps the same logic, but the scientific quality of the result depends on the label quality, sampling, survey bandpasses, and preprocessing used to build `real_lightcurves.csv`.
