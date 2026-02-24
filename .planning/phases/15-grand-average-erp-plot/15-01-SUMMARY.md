---
phase: 15-grand-average-erp-plot
plan: 01
subsystem: visualization
tags: [matplotlib, erp, p300, grand-average, eeg]

requires:
  - phase: 11-time-window-analysis
    provides: load_erp_timeseries() returning (N, 128, n_times) avg_erps
  - phase: 10-electrode-selection
    provides: get_parietal_indices() for EGI HydroCel-128 channel mapping
provides:
  - plot_grand_average_erp() function in run_modma_erp.py
  - Grand-average ERP figure (PNG 300 DPI + PDF vector)
affects: []

tech-stack:
  added: []
  patterns: [grand-average ERP visualization with SEM shading]

key-files:
  created:
    - outputs/out_phase15/grand_avg_erp.png
    - outputs/out_phase15/grand_avg_erp.pdf
  modified:
    - scripts/modma/run_modma_erp.py

key-decisions:
  - "Fz mapped to E5, Cz to E55, Pz to E62 via coordinate-distance fallback"
  - "Full epoch range -100ms to 500ms plotted (not 0-800ms as ROADMAP stated)"

patterns-established:
  - "Grand-average ERP plot: 3-row subplot (Fz/Cz/Pz), MDD red solid + HC blue dashed, SEM shading, P300 window annotation"

requirements-completed: [ERP-04]

duration: 8min
completed: 2026-02-24
---

# Phase 15 Plan 01: Grand-Average ERP Plot Summary

**Grand-average ERP waveform plot on Fz/Cz/Pz with MDD vs HC grouping, SEM shading, and P300 window annotation (250-500ms)**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-24T00:32:40Z
- **Completed:** 2026-02-24T00:40:39Z
- **Tasks:** 1
- **Files modified:** 1 (+ 2 generated outputs)

## Accomplishments
- Added plot_grand_average_erp() to run_modma_erp.py
- Generated grand_avg_erp.png (512 KB, 300 DPI) and grand_avg_erp.pdf (37 KB vector)
- 3-row subplot layout (Fz/Cz/Pz) with MDD red solid, HC blue dashed, +/-1 SEM shading
- P300 window (250-500ms) shaded gray with label; stimulus onset at 0ms marked

## Task Commits

1. **Task 1: Implement plot_grand_average_erp and generate figures** - a73dc34 (feat)

## Files Created/Modified
- scripts/modma/run_modma_erp.py - Added plot_grand_average_erp(), updated __main__
- outputs/out_phase15/grand_avg_erp.png - 300 DPI raster ERP figure (gitignored)
- outputs/out_phase15/grand_avg_erp.pdf - Vector ERP figure (gitignored)

## Decisions Made
- EGI channel mapping: Fz->E5 (idx=4), Cz->E55 (idx=54), Pz->E62 (idx=61)
- Plot covers -100ms to 500ms (full epoch); ROADMAP said 0-800ms but TMAX=0.5
- Y-axis positive up (standard Cartesian)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ERP-04 requirement satisfied
- Phase 16 (PRE-02, ARCH-01: pyprep + .npz cache) is next

---
*Phase: 15-grand-average-erp-plot*
*Completed: 2026-02-24*

## Self-Check: PASSED

- FOUND: outputs/out_phase15/grand_avg_erp.png
- FOUND: outputs/out_phase15/grand_avg_erp.pdf
- FOUND: 15-01-SUMMARY.md
- FOUND: commit a73dc34
