---
phase: 06-literature-informed-feature-expansion
plan: 02
subsystem: replication
tags: [eeg, cross-dataset, 15-dim, l1-logistic, tdbrain, modma, permutation-test]
requires:
  - phase: 06-literature-informed-feature-expansion
    plan: 01
    provides: "15-dim extract_features, build_feature_model_pipeline(use_l1=True)"
  - phase: 05-confirmatory-replication
    provides: "run_phase5_replication.py template, BA=0.500 baseline"
provides:
  - "run_phase6_replication.py for 15-dim cross-dataset replication"
  - "Feature distribution analysis (per-feature mean/std MODMA vs TDBRAIN)"
  - "Definitive negative result: 15-dim BA=0.500, p=1.0"
affects: [final-conclusions]
tech-stack:
  added: []
  patterns: [L1 cross-dataset replication, feature distribution comparison]
key-files:
  created: [run_phase6_replication.py]
  modified: []
key-decisions:
  - "15-dim L1 cross-dataset BA=0.500, p=1.0 -- same as Phase 5 5-dim baseline"
  - "Feature expansion does not improve cross-dataset generalization"
  - "TDBRAIN: 356 subjects (312 MDD, 47 HC), 1780 windows, 15 features"
patterns-established:
  - "Feature distribution analysis as standard cross-dataset diagnostic"
requirements-completed: [REP-01, VAL-01]
duration: 3min
completed: 2026-02-22
---

# Phase 6 Plan 02: Cross-Dataset Replication (15-dim) Summary

**15-dim L1 model trained on MODMA, tested on TDBRAIN: BA=0.500, p=1.0 -- no improvement over Phase 5 5-dim baseline**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-22T12:32:17Z
- **Completed:** 2026-02-22T12:35:44Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Cross-dataset replication with 15-dim literature-informed features
- Feature distribution analysis comparing MODMA vs TDBRAIN per-feature mean/std
- Permutation test confirms chance-level performance (p=1.0)
- Phase 5 baseline comparison: both 5-dim and 15-dim yield BA=0.500

## Task Commits

1. **Task 1: Cross-dataset replication with 15-dim features** - `646f56c` (feat)

## Files Created/Modified
- `run_phase6_replication.py` - 15-dim L1 cross-dataset replication (MODMA->TDBRAIN)

## Decisions Made
- 15-dim L1 cross-dataset BA=0.500 identical to Phase 5 5-dim baseline
- Feature expansion (PLV, coherence, beta power) does not improve generalization
- TDBRAIN loaded 356 subjects, 1780 windows with 15 features

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All phases complete: both 5-dim and 15-dim pipelines produce negative cross-dataset results
- Conclusion: EEG spectral/connectivity features with logistic regression do not generalize from MODMA to TDBRAIN

---
*Phase: 06-literature-informed-feature-expansion*
*Completed: 2026-02-22*

## Self-Check: PASSED
All files and commits verified.
