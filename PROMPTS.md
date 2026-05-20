# Prompts for coding agents

## Prompt 1: inspect project

You are working on a Python research MVP for astrophysical light-curve classification.

Context:
- The scientific goal is not just classification, but testing reliability of ML classification under incomplete / sparse / noisy light curves.
- Current code has a synthetic MVP:
  - generate synthetic light curves,
  - extract features,
  - degrade curves to 5/10/20/50 points,
  - compare baseline RandomForest trained on full curves with robust RandomForest trained on degraded curves,
  - save metrics and plots.
- Do not rewrite everything unless needed.
- Keep code simple, reproducible and suitable for a 10–20 page university practice report.
- Use Python, pandas, numpy, scikit-learn, astropy, matplotlib.
- Avoid seaborn.
- Outputs should go to outputs/tables and outputs/figures.

Task:
Inspect the project structure, run the synthetic MVP, fix any errors, and summarize:
1. whether it runs,
2. which outputs are produced,
3. what the first results suggest,
4. what should be implemented next.

## Prompt 2: add real data loader later

Add a new module for loading real public labelled light curves.

Requirements:
- Do not remove the synthetic MVP.
- Add a separate data-loading path so the synthetic experiment still works.
- Prefer a small, reproducible sample first.
- The output format must match the synthetic table:
  object_id, label, time, mag, mag_err, band
- Add clear comments explaining where the data comes from.
- Save raw files under data/raw and processed tables under data/processed.
- Add a new script scripts/02_run_real_data_mvp.py that reuses existing degradation, feature extraction and evaluation code.
- If a public source requires manual download, document the exact URL and expected folder layout in README.md.

## Prompt 3: improve evaluation

Improve evaluation for the light-curve classification MVP.

Add:
- per-class precision, recall, F1 table,
- confusion matrix for each n_points setting,
- calibration/reliability curve for baseline and robust models,
- Brier score for multiclass probabilities,
- a CSV table suitable for inclusion in a LaTeX report.

Do not over-engineer. Keep all outputs reproducible and saved under outputs/.
