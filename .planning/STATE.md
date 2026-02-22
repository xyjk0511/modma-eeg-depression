# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** Fix QC parameters to retain enough subjects for MDD vs HC classification
**Current focus:** Phase 6 — Literature-informed feature expansion (5→15 dims)

## Current Position

Phase: 6 of 6 (Literature-Informed Feature Expansion) -- Plan 1/2 COMPLETE
Status: 15-dim features implemented; internal MODMA CV done (5-dim BA=0.613 vs 15-dim BA=0.500)
Last activity: 2026-02-22 — Plan 06-01 executed (feature expansion + internal validation)

Progress: [█████████░] 95%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 5.4 min
- Total execution time: 0.63 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1-QC Repair | 2/2 | 17 min | 8.5 min |
| 3-Feature Opt | 2/2 | 10 min | 5 min |
| 5-Replication | 2/2 | 6 min | 3 min |
| 6-Feature Expansion | 1/2 | 5 min | 5 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: 4 phases derived from 12 requirements (QC -> Preprocessing -> Features -> Validation)
- 01-01: Per-window QC discards any window with >0 channels exceeding 200uV post-interpolation
- 01-01: Dynamic max_bad_channels (12% of n_channels) overrides CLI value
- 01-02: Per-window QC threshold changed from zero-tolerance to 12% to match Raw-level gate
- 01-02: Group retention stats (mdd_rate, hc_rate, diff) saved to metrics.json

- 03: Feature reduction 139→29 dims (PSD-rel + alpha-asym + theta/beta-ratio + riem-top2)
- 03: Fixed QC 200μV, drop LightGBM, SVM+Logistic only
- 03: Permutation pre-extract features, permute labels only, 1000 iterations

- 03-01: Inline Riemannian top-2 (3x3 cov central/temporal/parietal)
- 03-01: Alpha asymmetry generalized with region param and 10-20 fallback

- 03-02: Permutation loop calls run_simplified_cv per iteration (no re-extraction)
- 03-02: Removed lenient_mask/bad_amp_candidates; fixed 200uV + dynamic_max_bad directly

- 04: Parietal focus (theta/alpha/TBR) outperforms frontal (BA 0.558→0.613)
- 04: BA=0.613 meets threshold but p=0.135 — statistically not significant
- 04: Conclusion: negative/inconclusive; pre-register for confirmatory replication

- 05-01: TDBRAIN BDF file discovery uses multiple BIDS naming patterns with glob fallback
- 05-01: Spectrum sanity check runs on first subject only (diagnostic, not gate)
- 05-01: 10-20 channel filtering uses STANDARD_1020_REGION_MAP from existing pipeline

- 05-02: Cross-dataset replication on TDBRAIN: BA=0.500, p=1.0 — definitive negative
- 05-02: TDBRAIN uses BrainVision format (.vhdr), not BDF; adapter updated accordingly
- 05-02: 356 TDBRAIN subjects loaded (320 MDD + 47 HC - 11 no EEG), 1820 windows

- 06-01: 15-dim L1 BA=0.500 vs 5-dim L2 BA=0.613 — expansion did not improve MODMA internal CV
- 06-01: PLV via Hilbert on bandpass-filtered region-averaged signals
- 06-01: Coherence via scipy.signal.coherence on region-averaged signals
- 06-01: L1 penalty with saga solver for 15-dim model

### Pending Todos

Plan 06-02: Cross-dataset replication on TDBRAIN with 15-dim features.

### Blockers/Concerns

- Research flag: pyprep RANSAC requires highpass BEFORE detection (MODMA EDF DC offset)
- Research flag: ICLabel on EGI-128 montage needs empirical validation on 2-3 subjects

## Session Continuity

Last session: 2026-02-22 12:29
Stopped at: Completed 06-01-PLAN.md (15-dim feature expansion + internal validation)
Resume file: .planning/phases/06-literature-informed-feature-expansion/06-02-PLAN.md
