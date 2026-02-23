# Project Research Summary

**Project:** MODMA ERP Task-State MDD Classification — v2.0 Milestone
**Domain:** ERP-based MDD vs HC binary classification (dot-probe, 128-channel EGI, LOSO)
**Researched:** 2026-02-22 / 2026-02-23
**Confidence:** HIGH

---

## Executive Summary

The v2.0 milestone extends the existing run_modma_erp.py baseline (BA=0.670, hcue P300 mean amplitude, 128 dims, PCA(20), LOSO) by enriching P300 features, adding N200, fusing all three dot-probe conditions, and validating with a permutation test. The literature is clear: MDD shows reduced P300 amplitude, longer P300 latency, and altered N200 for emotional stimuli. Multi-condition fusion (hcue + fcue + scue) is the highest-value addition — ACM 2024 reports 79.84% accuracy when all ERP components and conditions are combined. The recommended feature set is P300 mean + peak + latency per channel, N200 mean per channel, and a scue-hcue contrast vector, all extractable with NumPy from existing avg-ERP arrays.

Zero new library installs are required. Every capability needed (epoching, feature extraction, permutation testing) is already present in MNE 1.11.0, NumPy 2.4.2, scikit-learn 1.8.0, and joblib 1.5.3. The architecture is purely additive: extend load_erp_features() to return richer features and a sub_id-keyed dict, add a fusion block in __main__, and add permutation_test_loso(). The existing run_loso() Pipeline (StandardScaler -> PCA(20) -> LogisticRegression) requires no changes.

The dominant risk is dimensionality explosion from multi-condition concatenation. Naive hstack of 3 conditions x 512 features = 1536 dims for N~50 subjects will overfit. PCA(20) inside the existing Pipeline is the mitigation. The second risk is permutation test validity: the test must permute subject-level labels (len(y) == n_subjects), not trial-level labels. Both risks have straightforward, low-cost fixes already identified.

---

## Key Findings

### Recommended Stack

No new installs needed. Use np.trapezoid (not np.trapz) for area-under-curve in NumPy 2.x. Use NumPy windowed argmax/argmin directly on evoked.get_data() — mne.Evoked.get_peak() returns one channel-time pair, not per-channel vectors. Keep the manual joblib permutation loop; sklearn.permutation_test_score with LeaveOneOut does not support subject-level BA aggregation.

**Core technologies:**
- MNE 1.11.0: ERP epoching, baseline correction, avg-ERP per subject — already used
- NumPy 2.4.2: P300/N200 window masking, mean/peak/latency/area extraction, condition hstack
- scikit-learn 1.8.0: LOSO Pipeline with StandardScaler -> PCA(20) -> LogisticRegression — unchanged
- joblib 1.5.3: Parallel permutation test loop — reuse pattern from resting-state pipeline

Do not add: autoreject (calibrated for resting-state windows, not ERP epochs), pyriemann (resting-state covariance features), neurokit2 (NumPy handles peak detection), deep learning (N=52 insufficient).

### Expected Features

**Must have (P1):**
- P300 peak amplitude per channel (250-500ms, win.max(axis=1)) — most replicated MDD biomarker
- P300 peak latency per channel (times[mask][win.argmax(axis=1)]) — MDD shows longer latency
- P300 area under curve per channel (np.trapezoid) — robust to noise
- N200 mean amplitude per channel (100-250ms) — early attentional bias marker
- Multi-condition concatenation (hcue + fcue + scue) — captures sad-cue hypersensitivity and happy-cue blunting
- Condition contrast scue - hcue — direct asymmetric emotional bias biomarker; free once multi-condition done

**Should have (P2 — after P1 validated):**
- ROI-averaged features at Fz/Cz/Pz — reduces 128 dims to ~5-10 with domain knowledge
- N200 peak amplitude per channel (win.min(axis=1))
- P300 topographic gradient (frontal/parietal ratio) — ~3 dims
- fcue - hcue contrast

**Defer (v2+):**
- Single-trial variability — requires keeping raw trial data in memory; architectural change
- Fusion with resting-state features — out of scope

### Architecture Approach

Additive extension of run_modma_erp.py. Build order: (1) extend load_erp_features(condition) to return richer P300 + N200 features keyed by sub_id; (2) add permutation_test_loso() and validate on single-condition first; (3) add fusion block in __main__ that intersects sub_ids and hstacks. Extract to erp_features.py / erp_evaluate.py only if the file exceeds ~300 lines.

**Major components:**
1. load_erp_features(condition) — MNE I/O + epoching + avg-ERP + feature extraction; returns dict[sub_id -> (feat_vec, label)]
2. Fusion block (inline __main__) — intersect sub_ids across 3 conditions, np.hstack, build aligned X and y
3. permutation_test_loso(X, y, ids, pipe, n_perm=1000) — shuffle subject labels, re-run LOSO, compute p-value; features loaded once outside loop
4. run_loso() — unchanged; sklearn Pipeline handles StandardScaler + PCA(20) + LR inside each fold

### Critical Pitfalls

1. **Multi-condition dimensionality explosion** — 3 conditions x 512 features = 1536 dims for N~50. PCA(20) inside Pipeline is the mitigation. Assert X_fused.shape[1] < 1600; verify BA does not drop below single-condition baseline.

2. **Permutation test at wrong level** — permuting trial-level y produces a too-narrow null distribution and falsely significant p-values. Assert len(y) == n_subjects before running. The ERP pipeline has one feature vector per subject — permuting y directly is correct.

3. **Multiple comparisons across conditions** — testing hcue/fcue/scue separately and reporting the best p-value inflates Type I error to ~14%. Pre-specify hcue as primary condition; apply Bonferroni (p<0.017) if all three are tested.

4. **Wrong P300/N200 window for this dataset** — dot-probe task may shift the P300 peak outside 250-500ms. Plot grand-average ERP before LOSO; verify positive deflection exists. One-time check, not leakage.

5. **Subject misalignment in fusion** — hstacking without checking sub_id alignment corrupts labels silently. Build dict[sub_id -> feat] per condition, intersect keys explicitly, align rows by sorted sub_id list.

---

## Implications for Roadmap

### Phase 1: P300 Feature Enrichment + N200

**Rationale:** Lowest risk, highest confidence. Extends existing working pipeline with additional features from the same avg-ERP arrays. No new data loading, no subject alignment complexity. Validates richer single-condition features before adding multi-condition complexity.
**Delivers:** Per-channel P300 peak amplitude, latency, area-under-curve; N200 mean amplitude. Single-condition (hcue) LOSO BA with enriched features.
**Addresses:** P1 features (P300 peak, latency, area; N200 mean)
**Avoids:** Pitfall 4 (wrong window) — verify grand-average ERP first; Pitfall 7 (N200/P300 correlation) — check feature correlation matrix

### Phase 2: Permutation Test Statistical Validation

**Rationale:** Validate single-condition results before adding multi-condition complexity. Permutation test logic is independent of fusion and should be verified on a known result (hcue BA=0.670) before being applied to fused features.
**Delivers:** Permutation p-value for hcue enriched features. Confirmed subject-level permutation logic. Reusable permutation_test_loso() function.
**Addresses:** Pitfall 6 (permutation at wrong level) — assert len(y) == n_subjects; Pitfall 5 (multiple comparisons) — pre-specify hcue as primary
**Uses:** joblib.Parallel pattern from resting-state pipeline

### Phase 3: Multi-Condition Fusion + Contrast Features

**Rationale:** Depends on Phase 1 (per-condition feature dicts with sub_id keys) and Phase 2 (permutation test ready). Highest potential BA gain but highest dimensionality risk — must come after single-condition validation.
**Delivers:** Fused X (hcue + fcue + scue), subject-aligned. Condition contrast (scue - hcue). LOSO BA + permutation p-value on fused features.
**Addresses:** Pitfall 3 (dimensionality) — PCA(20) inside Pipeline; Pitfall 5 (multiple comparisons) — Bonferroni; Anti-pattern 2 (subject misalignment) — explicit sub_id intersection

### Phase Ordering Rationale

- Phase 1 before Phase 3: feature extraction must be correct on single condition before fusion adds alignment complexity
- Phase 2 before Phase 3: permutation test logic must be verified on a known result before the more complex fused experiment
- Contrast features in Phase 3: requires both conditions feature dicts; natural byproduct of fusion work

### Research Flags

Standard patterns (skip deeper research):
- **Phase 1:** Well-documented NumPy windowed operations on avg-ERP arrays. MNE API confirmed.
- **Phase 2:** Permutation test pattern already exists in modma_mdd_real_experiment.py. Direct reuse.

Needs validation during implementation:
- **Phase 3:** Subject count after intersecting all three conditions is unknown. If n_common < 40, treat multi-condition as ablation rather than primary result.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new installs confirmed. All APIs verified against installed versions. np.trapezoid vs np.trapz confirmed for NumPy 2.x. |
| Features | MEDIUM | P300 amplitude/latency HIGH confidence (replicated, iSPOT-D n>1000). N200 LOW-MEDIUM (fewer replications). ACM 2024 multi-condition result MEDIUM (single paper, different dataset). |
| Architecture | HIGH | Based on direct inspection of run_modma_erp.py. Build order and component boundaries unambiguous. |
| Pitfalls | HIGH | Leakage pitfalls from Brookshire 2024 and Saarschmidt 2021. Permutation level pitfall from direct codebase analysis. |

**Overall confidence:** HIGH for implementation approach. MEDIUM for expected BA improvement.

### Gaps to Address

- **Trial counts per condition:** Unknown how many subjects have >=10 trials in fcue and scue. Check before Phase 3. If n_common < 40, single-condition hcue remains the primary result.
- **Grand-average ERP window verification:** P300 window (250-500ms) assumed valid for dot-probe task. Must plot grand-average ERP at start of Phase 1 to confirm.
- **N200 discriminability:** If N200 features do not improve BA in Phase 1 ablation, exclude from Phase 3 fusion to reduce dimensionality.

---

## Sources

### Primary (HIGH confidence)
- run_modma_erp.py — direct codebase inspection; current pipeline, LOSO, avg-ERP, BA=0.670 baseline
- Brookshire et al., Frontiers in Neuroscience 2024 — data leakage in EEG studies
- Saarschmidt et al., Scientific Reports 2021 — feature selection leakage inflates BA 10-20%
- MNE-Python 1.11.0 Epochs/Evoked API — epoching, baseline, get_data(), get_peak() limitation confirmed

### Secondary (MEDIUM confidence)
- ACM 2024 (Decoding Functional Brain Data for Emotion Recognition) — 79.84% with all ERP components + conditions
- Frontiers Human Neuroscience 2022 — frontal P300 latency as MDD biomarker
- Frontiers Psychiatry 2022 — P300 latency at C3/Pz discriminates MDD subtypes
- Frontiers Human Neuroscience 2020 — enhanced P3 to sad cues, reduced P1/P3 to happy cues in MDD dot-probe
- Nature Scientific Reports 2023 — frontal-parietal ERP differences in depressive states
- Nature Mental Health 2023 — technical considerations for EEG biomarkers in MDD

### Tertiary (LOW confidence)
- Biological Psychiatry 1993 — N200 in depression (foundational but old; limited replication)

---
*Research completed: 2026-02-23*
*Ready for roadmap: yes*
