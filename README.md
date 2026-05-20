# ML Light Curve Practice

Starter project for the scientific-practice MVP:

**Goal:** test how light-curve classification quality changes when curves become incomplete/sparse/noisy.

This starter version uses synthetic light curves only. It is a smoke test for the code structure:
- generate synthetic light curves for several astrophysical-like classes;
- extract interpretable features;
- train a baseline model on full curves;
- train a robust model on degraded curves;
- compare both models on degraded test sets;
- save plots and metric tables.

Next step: replace/add real public data loader (OGLE / ASAS-SN / Gaia-derived tables).

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
│   └── 01_run_synthetic_mvp.py
├── src/
│   └── lightcurve_ml/
│       ├── __init__.py
│       ├── degradation.py
│       ├── features.py
│       ├── plotting.py
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

## Interpretation

This synthetic MVP is not a final scientific result. It checks the experimental logic:
baseline trained on full data vs robust model trained on degraded data.

After this works, the same pipeline will be connected to public labelled light curves.
