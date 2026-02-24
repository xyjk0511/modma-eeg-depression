---
phase: 16-preprocessing-deferred
plan: 01
subsystem: preprocessing
tags: [pyprep, NoisyChannels, RANSAC, bad-channel-detection]

requires:
  - phase: 01-qc-repair
    provides: amplitude-based bad channel detection in load_windows()
provides:
  - pyprep NoisyChannels multi-criteria bad channel detection
  - adaptive exclusion threshold and A/B comparison runner
affects: [16-02, resting-state-pipeline]

tech-stack:
  added: [pyprep 0.5.0]
  patterns: [conditional detection with fallback, adaptive threshold]

key-files:
  modified: [scripts/modma/modma_mdd_real_experiment.py]

key-decisions:
  - "do_detrend=False since data already highpass at 1.0 Hz"
  - "Amplitude fallback preserves original 25% interpolation-only behavior"
  - "Regression gate uses resting-state BA=0.613 baseline"
  - "Cache version bumped 2->3 to invalidate pre-pyprep caches"

requirements-completed: [PRE-02]

duration: 9min
completed: 2026-02-24
---

# Phase 16 Plan 01: pyprep NoisyChannels Integration Summary

**pyprep NoisyChannels replaces amplitude-only bad channel detection with PREP-standard multi-criteria detection (deviation, correlation, HF noise, RANSAC), adaptive exclusion threshold, and A/B comparison runner**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-24T01:25:41Z
- **Completed:** 2026-02-24T01:34:46Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- pyprep NoisyChannels integrated into load_windows() with do_detrend=False
- Old amplitude detection preserved as fallback behind --no-use-pyprep
- Adaptive threshold loops 20%/25%/30% for >= 35 subject retention
- A/B comparison runner with structured table and JSON output
- Regression gate evaluates pyprep BA against resting-state baseline

## Task Commits

1. **Task 1: Integrate pyprep NoisyChannels into load_windows()** - `f86ca6a` (feat)
2. **Task 2: Adaptive threshold + A/B comparison runner** - `6799d80` (feat)

## Files Created/Modified
- `scripts/modma/modma_mdd_real_experiment.py` - pyprep integration, adaptive threshold, A/B runner

## Decisions Made
- do_detrend=False: data already highpass filtered at 1.0 Hz
- Amplitude fallback preserves original behavior (no exclusion, interpolation only)
- Regression gate targets resting-state BA=0.613, not ERP BA=0.670
- Cache version 2->3 invalidates old amplitude-based caches

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Amplitude fallback path exclusion behavior**
- **Found during:** Task 1
- **Issue:** Initial implementation applied exclusion to both paths, but original amplitude code never excluded subjects
- **Fix:** Split logic: pyprep excludes; amplitude preserves original interpolation-only behavior
- **Committed in:** f86ca6a

**Total deviations:** 1 auto-fixed (1 bug fix)

## Issues Encountered
None

## User Setup Required
None - pyprep 0.5.0 installed via pip.

## Next Phase Readiness
- Ready for Phase 16 Plan 02 (ERP .npz caching)
- Full A/B comparison on all 52 subjects recommended

---
*Phase: 16-preprocessing-deferred*
*Completed: 2026-02-24*

## Self-Check: PASSED
- Commit f86ca6a: FOUND
- Commit 6799d80: FOUND
- 16-01-SUMMARY.md: FOUND
- scripts/modma/modma_mdd_real_experiment.py: FOUND
