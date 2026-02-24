# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-23)

**Core value:** 深化 ERP 分析（更多成分、电极优化、时间窗口、特征解释），在 hcue P300 显著基线上进一步提升可解释性
**Current focus:** Gap closure — Phase 16 Preprocessing Deferred Items

## Current Position

Phase: 16 — Preprocessing Deferred Items
Plan: 16-02 (complete) — 2/2 plans done
Status: Complete
Last activity: 2026-02-24 — 16-02 ERP .npz caching complete (ARCH-01 closed)

Progress: [████████████████] 100%  (gap closure: 3/3 phases complete)

## v2.0 Final Results

| 实验 | N | BA | AUC | p-value | 结论 |
|------|---|-----|-----|---------|------|
| hcue 单条件 (128-dim) | 52 | 0.670 | 0.670 | 0.022 | ✓ 显著 |
| 三条件融合 (384-dim) | 51 | 0.521 | 0.508 | 0.411 | ✗ 负结果 |
| 对比 scue−hcue (128-dim) | 52 | 0.521 | 0.438 | 0.383 | ✗ 负结果 |

**结论：** hcue P300 均值幅度（128-dim）是最强信号，BA=0.670, p=0.022（统计显著）。
多条件融合引入噪声，性能下降至接近随机水平。

## Performance Metrics

**Velocity:**
- Total plans completed: 10
- Average duration: 4.9 min
- Total execution time: 0.75 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1-QC Repair | 2/2 | 17 min | 8.5 min |
| 3-Feature Opt | 2/2 | 10 min | 5 min |
| 5-Replication | 2/2 | 6 min | 3 min |
| 6-Feature Expansion | 2/2 | 8 min | 4 min |
| 16-Preprocessing Deferred | 2/2 | 13 min | 6.5 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- hcue BA=0.670 is statistically significant (p=0.022 < 0.05) — ERP-07/08/09 satisfied
- 09-01: Fusion (n=51) BA=0.521 p=0.412 — negative; hcue single-condition is optimal
- 09-01: Contrast scue-hcue (n=52) BA=0.521 p=0.384 — negative ablation
- 09-01: n_common=51 >= 40 threshold; ERP-10/11/12 satisfied
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
- [Phase 10]: 10-01: EGI HydroCel-128 uses E1-E128 naming; coordinate-distance fallback maps Pz->E62, P3->E60, P4->E85, Cz/CPz->E55
- [Phase 10]: 10-01: 5-dim parietal subset (BA=0.634, p=0.045) is significant; 3-dim (BA=0.521, p=0.430) is not
- [Phase 11]: 11-01: load_erp_timeseries retains avg_erp (128 x n_times) per subject alongside P300 feature vector
- [Phase 11]: 11-01: C=1.0 default preserves backward compat; bin ablation uses C=0.1 for 128-dim input
- [Phase 11]: 11-01: Half-open bin intervals [t0, t1) for bins 1-4; closed [t0, t1] for last bin to include 500ms

- [Phase 12]: 12-01: n_components=20 for both models (consistent with Phase 8, not Phase 11 capped value)
- [Phase 12]: 12-01: C=1.0 for hcue, C=0.1 for bin250; top channel E55 (Cz-adjacent) ranks #1 for both models with HC+ sign

- [Phase 14]: 14-01: Phase 02 status=passed (npz cache, pyprep deferred); Phase 05/07/13 status=negative
- [Phase 14]: 14-01: Traceability maps requirements to original implementing phase, not Phase 14

- [Phase 15]: 15-01: Fz->E5, Cz->E55, Pz->E62 via coordinate-distance fallback; plot covers -100ms to 500ms (full epoch)

- [Phase 16]: 16-01: pyprep NoisyChannels with do_detrend=False replaces amplitude-only detection in load_windows()
- [Phase 16]: 16-01: Amplitude fallback preserves original 25% interpolation-only behavior (no exclusion)
- [Phase 16]: 16-01: Adaptive threshold 20%/25%/30% ensures >= 35 subjects retained
- [Phase 16]: 16-01: Regression gate uses resting-state BA=0.613 baseline, not ERP BA=0.670

- [Phase 16]: 16-02: ERP .npz cache via SHA-256 hash key (version + params + file mtimes); --no-cache bypasses
- [Phase 16]: 16-02: Task 2 no-op — _CACHE_VERSION already "3" and cache key already includes use_pyprep/max_bad_pct from 16-01

### Pending Todos

Phase 16 complete (2/2 plans). All gap closure phases done.

### Blockers/Concerns

- Research flag: pyprep RANSAC requires highpass BEFORE detection (MODMA EDF DC offset) — RESOLVED in 16-01
- Research flag: ICLabel on EGI-128 montage needs empirical validation on 2-3 subjects

## Session Continuity

Last session: 2026-02-24
Stopped at: Completed 16-02-PLAN.md (ERP .npz caching, ARCH-01 closed, Phase 16 complete)
Resume file: .planning/ROADMAP.md — Phase 16 complete (2/2 plans)
