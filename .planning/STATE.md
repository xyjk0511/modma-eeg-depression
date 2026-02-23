# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** ERP 任务态特征（P300/N200）+ 多条件融合，实现统计显著的 MDD vs HC 分类
**Current focus:** Phase 7 — P300 特征丰富化 + N200 成分提取

## Current Position

Phase: 7 (ready to plan)
Plan: —
Status: Requirements + Roadmap 已定义，等待 /gsd:plan-phase 7
Last activity: 2026-02-23 — v2.0 里程碑 requirements + roadmap 完成

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 5.1 min
- Total execution time: 0.68 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1-QC Repair | 2/2 | 17 min | 8.5 min |
| 3-Feature Opt | 2/2 | 10 min | 5 min |
| 5-Replication | 2/2 | 6 min | 3 min |
| 6-Feature Expansion | 2/2 | 8 min | 4 min |

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

- 06-02: 15-dim L1 cross-dataset BA=0.500, p=1.0 — same as Phase 5 5-dim baseline
- 06-02: Feature expansion does not improve cross-dataset generalization
- 06-02: TDBRAIN: 356 subjects (312 MDD, 47 HC), 1780 windows, 15 features

### Pending Todos

None — all phases complete.

### Blockers/Concerns

- Research flag: pyprep RANSAC requires highpass BEFORE detection (MODMA EDF DC offset)
- Research flag: ICLabel on EGI-128 montage needs empirical validation on 2-3 subjects

## Session Continuity

Last session: 2026-02-22 12:35
Stopped at: Completed 06-02-PLAN.md (15-dim cross-dataset replication — ALL PHASES DONE)
Resume file: N/A — project complete
