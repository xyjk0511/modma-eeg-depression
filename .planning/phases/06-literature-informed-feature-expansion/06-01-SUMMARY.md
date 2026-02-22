---
phase: 06-literature-informed-feature-expansion
plan: 01
subsystem: features
tags: [eeg, plv, coherence, beta-power, logistic-regression, l1]
requires:
  - phase: 05-confirmatory-replication
    provides: "5-dim extract_features, locked pipeline, negative replication result"
provides:
  - "15-dim extract_features (backward compatible)"
  - "PLV and coherence connectivity helpers"
  - "run_phase6_validation.py for 5-dim vs 15-dim comparison"
  - "build_feature_model_pipeline with use_l1 param"
affects: [06-02, cross-dataset-replication]
tech-stack:
  added: [scipy.signal.hilbert, scipy.signal.sosfilt, scipy.signal.butter, scipy.signal.coherence]
  patterns: [PLV via Hilbert transform, coherence via scipy, L1/saga for high-dim logistic]
key-files:
  created: [run_phase6_validation.py]
  modified: [modma_mdd_real_experiment.py, external_data_adapter.py]
key-decisions:
  - "15-dim L1 BA=0.500 vs 5-dim L2 BA=0.613 -- expansion did not improve MODMA internal CV"
  - "PLV computed via Hilbert on bandpass-filtered region-averaged signals"
  - "Coherence via scipy.signal.coherence on region-averaged signals"
  - "L1 penalty with saga solver for 15-dim model to handle sparsity"
patterns-established:
  - "_compute_plv and _compute_coherence as reusable connectivity helpers"
requirements-completed: [FEAT-02, VAL-01]
duration: 5min
completed: 2026-02-22
---

# Phase 6 Plan 01: Feature Expansion + Internal Validation Summary

**15-dim feature expansion (beta power, PLV/coherence connectivity, temporal regions) with MODMA internal CV showing 5-dim BA=0.613 vs 15-dim BA=0.500**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-22T12:23:44Z
- **Completed:** 2026-02-22T12:28:44Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Expanded extract_features from 5 to 15 dimensions with full backward compatibility
- Added PLV and coherence connectivity features between brain regions
- Added beta power features (temporal/frontal beta2/beta3 rel, beta asymmetry)
- MODMA internal CV: 5-dim BA=0.613, 15-dim BA=0.500

## Task Commits

1. **Task 1: Expand extract_features from 5 to 15 dimensions** - `f90594c` (feat)
2. **Task 2: MODMA internal validation comparing 5-dim vs 15-dim** - `d1a8ba5` (feat)

## Files Created/Modified
- `modma_mdd_real_experiment.py` - 15-dim extract_features, _compute_plv, _compute_coherence, use_l1 param
- `external_data_adapter.py` - Updated self-test assertion from 5 to 15 features
- `run_phase6_validation.py` - 5-dim vs 15-dim MODMA internal CV with permutation test

## Decisions Made
- 15-dim L1/saga model underperforms 5-dim L2 on MODMA (BA 0.500 vs 0.613)
- PLV uses Hilbert transform on bandpass-filtered region-averaged signals
- Empty region indices return zeros instead of raising errors (robustness)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 15-dim features ready for cross-dataset replication on TDBRAIN (plan 06-02)
- Note: 15-dim did not improve MODMA internal CV

---
*Phase: 06-literature-informed-feature-expansion*
*Completed: 2026-02-22*

## Self-Check: PASSED
All files and commits verified.
