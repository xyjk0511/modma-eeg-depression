---
phase: 05-confirmatory-replication-with-pre-registered-pipeline
plan: 01
subsystem: validation
tags: [pre-registration, tdbrain, bdf, eeg, replication, data-adapter]

requires:
  - phase: 04-statistical-validation
    provides: "Locked 5-dim parietal pipeline (BA=0.613, p=0.135)"
provides:
  - "PRE_REGISTRATION.md with all locked pipeline parameters"
  - "external_data_adapter.py with TDBRAIN BDF loader and spectrum sanity check"
  - "Git tag pre-registration-v1 before any external data processing"
affects: [05-02-replication-run]

tech-stack:
  added: []
  patterns: ["data-adapter pattern for cross-dataset loading", "post-resample spectrum sanity check"]

key-files:
  created:
    - docs/PRE_REGISTRATION.md
    - external_data_adapter.py
  modified: []

key-decisions:
  - "TDBRAIN BDF file discovery uses multiple BIDS naming patterns with glob fallback"
  - "Spectrum sanity check runs on first subject only (diagnostic, not gate)"
  - "10-20 channel filtering uses STANDARD_1020_REGION_MAP from existing pipeline"

patterns-established:
  - "Data adapter pattern: thin loader returning same format as load_windows()"
  - "Pre-registration workflow: document + tag before any external data"

requirements-completed: [REP-01, REP-02]

duration: 3min
completed: 2026-02-22
---

# Phase 5 Plan 01: Pre-Registration + External Data Adapter Summary

**Pre-registration document and TDBRAIN data adapter with post-resample spectrum sanity check, git-tagged before any external data processing**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-22T10:09:30Z
- **Completed:** 2026-02-22T10:13:03Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- PRE_REGISTRATION.md documents all locked parameters (5-dim features, Logistic C=0.1, evaluation criteria)
- external_data_adapter.py loads TDBRAIN BDF files with 10-20 channel mapping and locked preprocessing
- Post-resample Welch PSD sanity check verifies alpha peak and detects Nyquist artifacts
- Self-test passes on synthetic dummy data (no real TDBRAIN data touched)
- Git tag `pre-registration-v1` created on commit c614296

## Task Commits

1. **Task 1: Pre-registration document + external data adapter** - `c614296` (feat)
2. **Task 2: Git-tag pre-registration** - tag `pre-registration-v1` on `c614296`

## Files Created/Modified
- `docs/PRE_REGISTRATION.md` - Complete pre-registration with locked pipeline, preprocessing, evaluation criteria
- `external_data_adapter.py` - TDBRAIN BDF loader with channel mapping, QC, spectrum sanity check, self-test

## Decisions Made
- TDBRAIN BDF file discovery tries multiple BIDS naming patterns (restEC, rest, generic) with glob fallback
- Spectrum sanity check is diagnostic-only (logs warnings, does not gate processing)
- Channel filtering uses existing STANDARD_1020_REGION_MAP — no new mapping code needed
- Participants.tsv parsing normalizes column names and maps common diagnosis labels to MDD/HC

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Pre-registration is locked and tagged — Plan 05-02 can proceed to download TDBRAIN and run replication
- external_data_adapter.py is ready to load real TDBRAIN data when available
- Blocker for 05-02: TDBRAIN data must be downloaded to local filesystem

## Self-Check: PASSED

- FOUND: 05-01-SUMMARY.md
- FOUND: commit c614296
- FOUND: external_data_adapter.py
- FOUND: docs/PRE_REGISTRATION.md
- FOUND: tag pre-registration-v1

---
*Phase: 05-confirmatory-replication-with-pre-registered-pipeline*
*Completed: 2026-02-22*
