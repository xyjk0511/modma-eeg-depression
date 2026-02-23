---
phase: 11-time-window-analysis
plan: 01
subsystem: analysis
tags: [eeg, erp, p300, t-test, fdr, permutation, ablation, matplotlib]

requires:
  - phase: 10-electrode-selection
    provides: get_parietal_indices, _loso_ba_subset, permutation_test_subset

provides:
  - load_erp_timeseries (avg_erp timeseries per subject, shape N x 128 x n_times)
  - plot_ttest_curve (out_phase11/ttest_curve.png)
  - run_bin_ablation (out_phase11/bin_ablation.csv)
  - run_time_window_analysis (Phase 11 entry point)

affects: [future-phases, reporting]

tech-stack:
  added: [scipy.stats.ttest_ind, scipy.stats.false_discovery_control, csv, matplotlib]
  patterns: [half-open bin intervals, FDR correction, C param forwarded through permutation stack]

key-files:
  created:
    - out_phase11/ttest_curve.png
    - out_phase11/bin_ablation.csv
  modified:
    - run_modma_erp.py

key-decisions:
  - "load_erp_timeseries retains avg_erp per subject (128 x n_times) alongside P300 feature vector"
  - "C=1.0 default preserves backward compatibility for existing permutation_test_subset callers"
  - "Bin ablation uses C=0.1 for 128-dim full-channel classification"
  - "Half-open bin intervals [t0, t1) except last bin uses closed [t0, t1] to include 500ms"
  - "false_discovery_control returns adjusted p-values; sig_mask = adj_p < 0.05"

requirements-completed: [TWIN-01, TWIN-02]

duration: 17min
completed: 2026-02-23
---

# Phase 11 Plan 01: Time-Window Analysis Summary

**Per-timepoint FDR t-test curve (250-500ms parietal) + 5-bin 50ms ablation classification identifying peak discriminative P300 window**

## Performance

- **Duration:** 17 min
- **Started:** 2026-02-23T07:17:11Z
- **Completed:** 2026-02-23T07:34:36Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added load_erp_timeseries() returning per-subject avg-ERP array (N, 128, n_times)
- Added C parameter to _loso_ba_subset and permutation_test_subset (backward compatible)
- TWIN-01: plot_ttest_curve() with FDR correction, saved to out_phase11/ttest_curve.png
- TWIN-02: run_bin_ablation() 5 x 50ms bins, saved to out_phase11/bin_ablation.csv

## Task Commits

1. **Task 1: load_erp_timeseries + C param** - 38ed38b (feat)
2. **Task 2: plot_ttest_curve, run_bin_ablation, run_time_window_analysis** - b2e4ba3 (feat)

## Files Created/Modified

- run_modma_erp.py - Added load_erp_timeseries, plot_ttest_curve, run_bin_ablation, run_time_window_analysis, BINS, csv/scipy imports
- out_phase11/ttest_curve.png - t-stat vs time (250-500ms), FDR-sig points red, peak orange
- out_phase11/bin_ablation.csv - 5 rows (one per 50ms bin) with BA, p-value, Sig, Note

## Decisions Made

- load_erp_timeseries reuses per-subject loop from load_erp_features but retains avg_erp
- C=1.0 default preserves backward compatibility; bin ablation passes C=0.1 explicitly
- Half-open intervals [t0, t1) for bins 1-4, closed [t0, t1] for last bin

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Phase 11 complete; out_phase11/ttest_curve.png and out_phase11/bin_ablation.csv ready for reporting
- Peak discriminative bin identified from ablation table (best BA marked in CSV Note column)

---
*Phase: 11-time-window-analysis*
*Completed: 2026-02-23*
