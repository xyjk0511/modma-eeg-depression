---
phase: 03-feature-permutation-opt
plan: 01
subsystem: features
tags: [psd, theta-beta-ratio, riemannian, svm, logistic, eeg]
requires:
  - phase: 01-qc-repair
    provides: "QC-repaired windows with 43 subjects retained"
provides:
  - "29-dim extract_features (PSD-rel + alpha-asym + TBR + riem-top2)"
  - "run_simplified_cv for pre-extracted feature matrix"
affects: [03-02, 04-statistical-validation]
tech-stack:
  added: []
  patterns: [pre-extracted-features-cv, theta-beta-ratio-per-region]
key-files:
  created: []
  modified: [modma_mdd_real_experiment.py]
key-decisions:
  - "Inline Riemannian top-2 (3x3 cov central/temporal/parietal)"
  - "Alpha asymmetry generalized with region param and 10-20 fallback"
patterns-established:
  - "Feature extraction returns fixed 29-dim vector"
  - "CV operates on pre-extracted features, no re-extraction per fold"
requirements-completed: [FEAT-01]
duration: 6min
completed: 2026-02-22
---

# Phase 3 Plan 1: Feature Reduction & Simplified CV Summary

**29-dim feature extraction (PSD-rel, alpha-asymmetry, theta/beta ratio, Riemannian top-2) with pre-extracted-features CV using SVM+Logistic only**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-22T06:59:09Z
- **Completed:** 2026-02-22T07:05:10Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Reduced features from 139 to exactly 29 dimensions
- Added Theta/Beta ratio per region (5 dims, FEAT-01) with epsilon guard
- Created run_simplified_cv operating on pre-extracted features
- Removed 434 lines of dead code (DE, Hjorth, connectivity, lateral asymmetry, old CV)
- Simplified pipeline to SVM+Logistic only, no PCA/SelectKBest/LightGBM

## Task Commits

1. **Task 1: Rewrite extract_features to 29 dims** - `f2bd2f3` (feat)
2. **Task 2: Create run_simplified_cv** - `c6f4272` (feat)

## Files Created/Modified
- `modma_mdd_real_experiment.py` - Rewritten extract_features, new run_simplified_cv, simplified pipeline

## Decisions Made
- Inlined Riemannian using 3x3 cov (central/temporal/parietal) instead of full 5-region
- Generalized _alpha_asymmetry with region param; added 10-20 temporal fallback

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None

## Next Phase Readiness
- 29-dim extract_features ready for plan 03-02 (permutation loop rewrite)
- run_simplified_cv ready for rewritten permutation test
- build_report and run_main_with_output_dir still reference old functions (03-02 scope)

---
*Phase: 03-feature-permutation-opt*
*Completed: 2026-02-22*

## Self-Check: PASSED
- 03-01-SUMMARY.md: FOUND
- f2bd2f3 (Task 1): FOUND
- c6f4272 (Task 2): FOUND
