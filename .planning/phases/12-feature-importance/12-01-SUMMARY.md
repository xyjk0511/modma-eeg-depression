---
phase: 12-feature-importance
plan: 01
subsystem: analysis
tags: [mne, sklearn, pca, logistic-regression, loso, topomap, feature-importance]

requires:
  - phase: 11-time-window-analysis
    provides: load_erp_timeseries, bin250 C=0.1 convention
  - phase: 08-erp-classification
    provides: hcue 128-dim P300 features, C=1.0 baseline

provides:
  - _loso_collect_coefs() per-fold PCA back-projection to 128-channel weights
  - run_feature_importance() ranked weight tables + CSVs + side-by-side topomap
  - out_phase12/feature_importance_hcue.csv (128 rows)
  - out_phase12/feature_importance_bin250.csv (128 rows)
  - out_phase12/topomap_hcue_vs_bin.png

affects: [future interpretability phases, reporting]

tech-stack:
  added: [pandas (already installed, now imported)]
  patterns: [PCA back-projection via pca.components_.T @ clf.coef_[0], MNE Info-based topomap with names parameter for top-5 labels]

key-files:
  created:
    - out_phase12/feature_importance_hcue.csv
    - out_phase12/feature_importance_bin250.csv
    - out_phase12/topomap_hcue_vs_bin.png
  modified:
    - run_modma_erp.py

key-decisions:
  - "n_components=20 for both models (consistent with Phase 8 baseline)"
  - "C=1.0 for hcue full-window, C=0.1 for bin250 (matches respective training configs)"
  - "avg_abs = mean of |per-fold back-projected weights| for ranking; avg_weights = signed mean for topomap color"
  - "vlim=(-vmax, vmax) symmetric colorbar"

patterns-established:
  - "Back-projection: pca.components_.T @ clf.coef_[0] per fold, then average across folds"
  - "names_list=[''] * 128 with top-5 filled for MNE plot_topomap annotation"

requirements-completed: [FIMP-01]

duration: 16min
completed: 2026-02-23
---

# Phase 12 Plan 01: Feature Importance Summary

**LOSO PCA back-projection (pca.components_.T @ coef_[0]) for hcue and bin250 models, producing 128-channel ranked weight CSVs and side-by-side RdBu_r topomap**

## Performance

- **Duration:** 16 min
- **Started:** 2026-02-23T09:05:16Z
- **Completed:** 2026-02-23T09:21:39Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `_loso_collect_coefs()` — re-runs LOSO collecting per-fold back-projected channel weights (n_folds, 128)
- Added `run_feature_importance()` — produces ranked tables, CSVs, and side-by-side topomap for hcue and bin250
- Top channel E55 (Cz-adjacent) ranks #1 for both models with HC+ sign; E31 and E107 also prominent

## Task Commits

1. **Task 1+2: Add _loso_collect_coefs and run_feature_importance** - `6acc13e` (feat)

## Files Created/Modified

- `run_modma_erp.py` - Added `import pandas as pd`, `_loso_collect_coefs()`, `run_feature_importance()`, call in `__main__`
- `out_phase12/feature_importance_hcue.csv` - 128-row ranked weights for hcue model (top: E55 w=0.7281)
- `out_phase12/feature_importance_bin250.csv` - 128-row ranked weights for bin250 model (top: E55 w=0.3027)
- `out_phase12/topomap_hcue_vs_bin.png` - Side-by-side RdBu_r topomaps, 348KB

## Decisions Made

- n_components=20 for both models (consistent with Phase 8, not Phase 11 capped value of 51)
- C=1.0 for hcue, C=0.1 for bin250 — matches training configs that produced reported BA values
- Ranking by mean(|per-fold weights|) across folds; signed mean used for topomap color direction

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Feature importance outputs ready for reporting/interpretation
- Top channels (E55, E31, E107, E114) are central/parietal — consistent with P300 spatial distribution
- v3.0 milestone complete (Phase 12 was final phase)

---
*Phase: 12-feature-importance*
*Completed: 2026-02-23*
