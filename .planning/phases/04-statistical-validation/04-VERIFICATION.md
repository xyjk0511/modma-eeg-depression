# Phase 4: Statistical Validation — Verification Report

**Date:** 2026-02-22
**Conclusion:** NEGATIVE (performance meets threshold, statistically not significant)

## Results Summary

| Metric | Value | Criterion | Pass |
|--------|-------|-----------|------|
| Balanced Accuracy | 0.613 | > 0.60 | YES |
| CI Lower (95%) | 0.466 | > 0.50 | NO |
| Permutation p-value | 0.135 | < 0.05 | NO |
| Min subjects/class | 18 | >= 5 | YES |

**Overall:** 2/4 criteria met → negative conclusion.

## Feature Variant Diagnostic (8 configs tested)

| Config | Dims | BA | Notes |
|--------|------|----|-------|
| B_parietal_focus | 5 | 0.613 | **Best — selected** |
| E_multi_theta | 6 | 0.593 | |
| C_add_hjorth | 7 | 0.570 | |
| A_baseline_5dim | 5 | 0.558 | Phase 3 baseline |
| D_add_temporal_asym | 6 | 0.535 | |
| G_add_riem_cp | 6 | 0.535 | |
| H_delta_frontal | 5 | 0.523 | |
| F_minimal_3dim | 3 | 0.500 | |

## Final Pipeline

- **Features (5-dim):** parietal_theta_rel, parietal_alpha_rel, alpha_asym_frontal, parietal_TBR, riem_central_temporal
- **Model:** LogisticRegression(C=0.1, class_weight="balanced")
- **CV:** StratifiedGroupKFold (subject-level, no data leakage)
- **Permutations:** 1000 (label permutation, pre-extracted features)
- **Subjects:** 43 (18 MDD, 25 HC after QC)

## Interpretation

BA=0.613 exceeds the 0.60 threshold, indicating the parietal-focused feature set captures some MDD-related signal. However:

1. **CI includes chance level** (0.466–0.756): the true BA could be at or below 0.50
2. **p=0.135**: ~13.5% of permuted models achieve BA >= 0.613, insufficient to reject H0
3. **Root cause:** small sample size (N=43) limits statistical power

This is an **exploratory** result — the pipeline shows promise but cannot be declared significant.

## Follow-up Actions

1. **Pre-register the fixed pipeline** (5-dim parietal + Logistic C=0.1) for a confirmatory replication on held-out or new data
2. **Expand sample size** or use an external validation dataset before declaring the model "usable"

## Artifacts

- `results_phase4_final/metrics.json` — full metrics with conclusion object
- `results_phase4_final/roc_curve.png` — ROC curve
- `results_phase4_final/confusion_matrix.png` — confusion matrix
- `results_phase4_final/qc_report.csv` — subject-level QC
- `run_phase4_diag.py` — diagnostic script (8 configs)
- `run_phase4_val.py` — full validation script (1000 perms)
