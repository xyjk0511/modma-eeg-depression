---
phase: 01-qc-repair
plan: 01
subsystem: preprocessing
tags: [mne, interpolation, eeg, qc, montage, highpass]

requires: []
provides:
  - "Montage-based bad channel interpolation in load_windows"
  - "Dynamic max_bad_channels (12% of channel count)"
  - "Window 0 skip for filter transient avoidance"
  - "Per-window post-interpolation QC gate"
  - "Highpass default raised to 1.0 Hz"
affects: [01-qc-repair, 02-preprocessing]

tech-stack:
  added: []
  patterns: ["Raw-level interpolation before average reference", "Per-window post-interpolation QC"]

key-files:
  created: []
  modified: ["modma_mdd_real_experiment.py"]

key-decisions:
  - "Per-window QC discards any window with >0 channels exceeding 200uV post-interpolation"
  - "Dynamic max_bad_channels computed as int(n_channels * 0.12) overrides CLI value"

patterns-established:
  - "Preprocessing order: load -> montage -> crop -> highpass -> detect bad -> interpolate -> reref -> resample -> window"
  - "Window 0 always skipped to avoid filter transient"

requirements-completed: [QC-01, QC-02, QC-03, PRE-01]

duration: 3min
completed: 2026-02-22
---

# Phase 1 Plan 1: QC Pipeline Repair Summary

**Montage-based spherical spline interpolation with W0 skip, 1.0Hz highpass, and dynamic 12% max_bad_channels gate**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-22T03:01:50Z
- **Completed:** 2026-02-22T03:04:47Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Restructured load_windows: montage -> crop -> highpass -> detect bad -> interpolate -> reref -> resample -> window
- Bad channels interpolated via MNE spherical spline (<=25% bad)
- Window 0 skipped for all subjects to avoid filter transient
- Per-window post-interpolation QC discards windows with any channel exceeding threshold
- Dynamic max_bad_channels (12% of n_channels) wired through all downstream calls

## Task Commits

1. **Task 1: Update parse_args and load_windows with interpolation pipeline** - `21b56ad` (feat)
2. **Task 2: Wire dynamic max_bad_channels through run_main_with_output_dir** - `56a5570` (feat)

## Files Created/Modified
- `modma_mdd_real_experiment.py` - Interpolation pipeline in load_windows, dynamic max_bad in run_main

## Decisions Made
- Per-window QC: any channel exceeding 200uV post-interpolation causes window discard
- Dynamic max_bad_channels overrides CLI --max-bad-channels; CLI arg kept for compat

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None

## Next Phase Readiness
- Interpolation pipeline ready for end-to-end testing (Plan 02)
- interp_info metadata available for QC report extension
- Expected retention: 43/53 subjects (18 MDD + 25 HC)

---
*Phase: 01-qc-repair*
*Completed: 2026-02-22*

## Self-Check: PASSED
All files and commits verified.
