# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** Fix QC parameters to retain enough subjects for MDD vs HC classification
**Current focus:** Project complete (all 4 phases done)

## Current Position

Phase: 4 of 4 (Statistical Validation) -- COMPLETE
Result: NEGATIVE — BA=0.613 (>0.60) but p=0.135 (>0.05)
Status: Performance meets threshold but statistically not significant
Last activity: 2026-02-22 — Phase 4 validation complete

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 6.8 min
- Total execution time: 0.45 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1-QC Repair | 2/2 | 17 min | 8.5 min |
| 3-Feature Opt | 2/2 | 10 min | 5 min |

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

### Pending Todos

None — project complete.

### Blockers/Concerns

- Research flag: pyprep RANSAC requires highpass BEFORE detection (MODMA EDF DC offset)
- Research flag: ICLabel on EGI-128 montage needs empirical validation on 2-3 subjects

## Session Continuity

Last session: 2026-02-22 07:14
Stopped at: Completed 03-02-PLAN.md (Phase 3 complete)
Resume file: .planning/phases/04-statistical-validation/04-01-PLAN.md
