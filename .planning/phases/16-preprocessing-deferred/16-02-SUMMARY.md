---
phase: 16-preprocessing-deferred
plan: 02
subsystem: preprocessing
tags: [npz-cache, ERP, load_erp_features, load_erp_timeseries]

requires:
  - phase: 16-preprocessing-deferred
    provides: pyprep NoisyChannels integration and _CACHE_VERSION bump to 3
provides:
  - ERP .npz caching for load_erp_features and load_erp_timeseries
  - --no-cache flag to bypass ERP cache
affects: [erp-pipeline, run_modma_erp]

tech-stack:
  added: []
  patterns: [hash-based npz cache with mtime invalidation]

key-files:
  modified: [scripts/modma/run_modma_erp.py]

key-decisions:
  - "Cache key SHA-256 includes version + condition + func_name + filter params + file mtimes"
  - "Task 2 no-op: _CACHE_VERSION already 3 and cache key includes use_pyprep/max_bad_pct from 16-01"

requirements-completed: [ARCH-01]

duration: 4min
completed: 2026-02-24
---

# Phase 16 Plan 02: ERP .npz Caching Summary

**ERP load functions cache to .npz via SHA-256 hash key (version + params + file mtimes); second run loads in <1s; --no-cache bypasses**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-24T01:39:13Z
- **Completed:** 2026-02-24T01:42:45Z
- **Tasks:** 2 (1 code change + 1 verified no-op)
- **Files modified:** 1

## Accomplishments
- _erp_cache_path() hashes version + params + sorted file mtimes to 16-char hex key
- load_erp_features() and load_erp_timeseries() check/save .npz cache
- set_erp_cache(False) via --no-cache flag bypasses cache
- Confirmed _CACHE_VERSION already "3" and cache key includes use_pyprep/max_bad_pct

## Task Commits

1. **Task 1: Add .npz cache to ERP loading functions** - `8ded045` (feat)
2. **Task 2: Bump resting-state cache version** - no-op (already done in 16-01)

## Files Created/Modified
- `scripts/modma/run_modma_erp.py` - ERP .npz caching, _erp_cache_path helper, set_erp_cache, --no-cache flag

## Decisions Made
- Cache key uses SHA-256 of version + condition + func_name + filter params + sorted file mtimes
- Task 2 confirmed as no-op: _CACHE_VERSION = "3" and use_pyprep/max_bad_pct in cache key were both done in Plan 16-01

## Deviations from Plan

Task 2 required no code changes. The plan specified bumping _CACHE_VERSION from "2" to "3" and verifying use_pyprep/max_bad_pct in cache key. Both were already completed in Plan 16-01 (commits f86ca6a and 6799d80).

**Total deviations:** 0 auto-fixed. 1 task was pre-satisfied.

## Issues Encountered
None

## User Setup Required
None

## Next Phase Readiness
- ERP caching complete, ARCH-01 closed
- Phase 16 fully complete (2/2 plans done)

---
*Phase: 16-preprocessing-deferred*
*Completed: 2026-02-24*

## Self-Check: PASSED
- Commit 8ded045: FOUND
- 16-02-SUMMARY.md: FOUND
- scripts/modma/run_modma_erp.py: FOUND
