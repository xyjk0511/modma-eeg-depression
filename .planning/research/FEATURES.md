# Feature Research

**Domain:** ERP task-state MDD vs HC binary classification (dot-probe hcue/fcue/scue, 128-channel, MODMA)
**Researched:** 2026-02-23
**Confidence:** MEDIUM (multiple web sources + official literature)

## Context: What Already Exists (v2.0 Complete)

| Feature | Status | Result |
|---------|--------|--------|
| hcue P300 mean amplitude per channel (128-dim, 250-500ms) | BUILT | BA=0.670, p=0.022 significant baseline |
| P300 peak amplitude + latency + AUC + N200 mean (640-dim) | BUILT, REVERTED | BA=0.554 regressed vs baseline |
| Multi-condition fusion hcue+fcue+scue (384-dim) | BUILT | BA=0.521, p=0.412 degraded |
| Contrast scue-hcue (128-dim) | BUILT | BA=0.521, p=0.384 degraded |
| LOSO cross-validation + permutation test (1000 perms) | BUILT | Infrastructure complete |

Key lesson from v2.0: Adding more features (640-dim, 384-dim) consistently degrades performance vs 128-dim baseline. N=52 subjects with PCA(20) means the curse of dimensionality is the dominant constraint. The winning strategy for v3.0 is dimensionality REDUCTION (electrode subset) and INTERPRETATION (which features drive BA=0.670).

---

## Feature Landscape: v3.0 NEW Features Only

### Table Stakes (Expected in Any ERP Deep Analysis)

| Feature | Why Expected | Complexity | Notes | Depends On |
|---------|--------------|------------|-------|------------|
| Electrode subset selection (Pz/P3/P4 parietal ROI) | P300 is maximal at parietal sites. Reducing 128 channels to 3-5 parietal electrodes cuts dimensionality 25x, directly addressing the curse-of-dimensionality that caused 640-dim regression. arXiv 2510.21969 uses {Fz, Pz, P3, P4, Oz} for small-sample P300 classification. | LOW | avg_erp[parietal_idx, p300_mask].mean(axis=1) shape (3,). No PCA needed at 3-dim. | Existing 128-dim pipeline; requires EGI montage channel lookup |
| Time-window t-test analysis (P300 latency band) | Identifies which 4ms time bins within 250-500ms are most discriminative between MDD and HC. Standard in ERP clinical literature. Provides scientific insight into whether MDD effect is early (250-350ms) or late (400-500ms). | LOW | scipy.stats.ttest_ind per time bin across subjects. Plot t-statistic vs time. No classifier needed. | Existing epoch data (avg_erp per subject) |
| LR coef_ interpretation (per-channel weights) | LogisticRegression coef_ directly gives per-feature weights. For 128-dim P300 mean features, maps back to channel space showing which channels drive BA=0.670. Already available in existing pipeline with zero model changes. | LOW | Average coef_ across LOSO folds. Plot as topographic map or bar chart. | Existing LOSO pipeline (LR already used) |
| Permutation importance (model-agnostic) | Shuffles one feature at a time, measures BA drop. Corrects for correlated features where raw coef_ can be misleading. Identifies which of the 128 channels are load-bearing vs noise. | MEDIUM | sklearn.inspection.permutation_importance inside LOSO loop on test fold. Aggregate across folds. | Existing LOSO pipeline |

### Differentiators (Can Provide Scientific Insight Beyond BA=0.670)

| Feature | Value Proposition | Complexity | Notes | Depends On |
|---------|-------------------|------------|-------|------------|
| Parietal-only classifier (3-dim: Pz/P3/P4 mean) | If 3 parietal channels achieve BA >= 0.670, proves signal is spatially concentrated and clinically interpretable. Eliminates need for PCA. Directly publishable finding. | LOW | feat = avg_erp[parietal_idx, p300_mask].mean(axis=1) shape (3,). StandardScaler + LR(C=1.0). No PCA. | Electrode subset selection (channel index lookup) |
| Time-window ablation (50ms bins) | Splits 250-500ms into 5 x 50ms bins, runs LOSO on each bin mean amplitude. Identifies peak discriminative window with classifier-based evidence (complements t-test). | LOW | Loop over [(250,300),(300,350),(350,400),(400,450),(450,500)]. Each bin: 128-dim mean -> LOSO BA. | Existing pipeline |
| Frontal vs parietal ROI comparison | Tests whether frontal (Fz, F3, F4) or parietal (Pz, P3, P4) channels carry more MDD signal. Informs electrode placement for future clinical tools. | LOW | Run LOSO separately on frontal-ROI and parietal-ROI feature vectors. Compare BA. | Electrode subset selection |
| N200 parietal-only (6-dim: P300+N200 at Pz/P3/P4) | N200 at parietal sites for emotional stimuli. If parietal N200 adds signal beyond P300, concatenate [P300_parietal, N200_parietal] = 6-dim. Low dimensionality avoids the regression seen with 640-dim. | LOW | Same extraction as P300 parietal but N200_WIN=(0.10, 0.25). Test 6-dim vs 3-dim. | Electrode subset selection |

### Anti-Features (Confirmed Problematic by v2.0 Results)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full 640-dim enriched features (P300 peak + latency + AUC + N200 mean) | Maximum information | Phase 7 proved this: BA dropped from 0.670 to 0.554. PCA(20) on 640-dim loses the spatial structure that makes 128-dim work. | Parietal ROI (3-dim) same information, no dimensionality problem. |
| Multi-condition fusion (384-dim) | More signal from emotional conditions | Phase 9 proved this: BA dropped from 0.670 to 0.521. fcue/scue add noise, not signal. | hcue single-condition only. |
| Condition contrast scue-hcue | Direct emotional bias biomarker | Phase 9 proved this: BA=0.521. Difference signal is noisier than raw hcue. | hcue single-condition only. |
| LPP component (500-800ms) | Theoretically captures sustained emotional processing | medrxiv 2020 found NO evidence for reduced LPP in MDD. Low prior probability. | P300 window (250-500ms) already captures the relevant signal. |
| Deep learning on ERP waveforms | Papers report high accuracy | N=52 is far too small for cross-subject generalization. | LR/SVM with low-dim handcrafted features. |
| Per-trial classification | More training samples | Trial-level accuracy is inflated (within-subject correlation). | Subject-level LOSO only. |

---

## Feature Dependencies



Dependency Notes:
- Electrode subset selection is the prerequisite for all ROI-based features. Requires EGI montage channel lookup (see Implementation Note).
- Time-window t-test is independent of the classifier. Can run in parallel with electrode selection work.
- LR coef_ is already available in the existing pipeline with zero model changes. Lowest-cost interpretation feature.
- Permutation importance must be computed inside the LOSO loop on the test fold, not after. Requires architectural change to run_loso().
- N200 parietal-only depends on electrode subset selection being done first.

---

## MVP Definition

### Launch With (v3.0 Phase 10) -- Electrode Subset + Interpretation

- [ ] Identify Pz/P3/P4 channel indices in 128-ch EGI montage -- prerequisite for all ROI features
- [ ] Parietal-only classifier (3-dim) -- tests if dimensionality reduction recovers or exceeds BA=0.670
- [ ] Time-window t-test (per 4ms bin, 250-500ms) -- identifies peak discriminative window, scientific insight
- [ ] LR coef_ interpretation across LOSO folds -- shows which channels drive BA=0.670, zero implementation cost

### Add After Validation (v3.1)

- [ ] Time-window ablation (50ms bins) -- trigger: parietal ROI validates, want to know if sub-window improves further
- [ ] Permutation importance (sklearn) -- trigger: LR coef_ shows unexpected channels, need model-agnostic confirmation
- [ ] Frontal vs parietal ROI comparison -- trigger: interpretation shows frontal channels are important

### Future Consideration (v3.2+)

- [ ] N200 parietal-only (6-dim) -- defer: only worth testing if 3-dim parietal P300 achieves BA >= 0.670
- [ ] LPP 500-800ms -- defer: medrxiv 2020 found no evidence for reduced LPP in MDD; low prior probability

---

## Feature Prioritization Matrix

| Feature | Classification Value | Implementation Cost | Priority |
|---------|---------------------|---------------------|----------|
| Parietal ROI 3-dim (Pz/P3/P4) | HIGH -- directly addresses dimensionality root cause | LOW | P1 |
| Time-window t-test | MEDIUM -- scientific insight, not classifier improvement | LOW | P1 |
| LR coef_ interpretation | MEDIUM -- interpretability, not BA improvement | LOW | P1 |
| Time-window ablation (50ms bins) | MEDIUM -- may find tighter window | LOW | P2 |
| Permutation importance | MEDIUM -- model-agnostic confirmation | MEDIUM | P2 |
| Frontal vs parietal comparison | LOW -- scientific, not performance | LOW | P2 |
| N200 parietal 6-dim | LOW -- v2.0 showed N200 does not help at 128-dim | LOW | P3 |
| LPP 500-800ms | LOW -- literature contradicts effect in MDD | LOW | P3 |

Priority key:
- P1: Must have -- directly addresses v3.0 milestone goal
- P2: Should have -- adds interpretability after P1 validated
- P3: Nice to have -- low prior probability based on v2.0 results and literature

---

## Critical Implementation Note: EGI 128-Channel Montage

The MODMA data uses EGI 128-channel Hydrocel montage. Channel names are numeric (1-128) or EGI-specific labels, NOT standard 10-20 names. Pz/P3/P4 must be identified by:

1. raw.get_montage() to get channel positions
2. Match by spatial coordinates to standard 10-20 positions
3. Or use mne.channels.make_standard_montage("GSN-HydroCel-128") and find nearest channels to Pz/P3/P4 coordinates

This is a non-trivial lookup step that must be done before any ROI feature extraction. Failure to correctly identify parietal channels will silently produce wrong features with no error.

---

## Literature Benchmarks

| Approach | Reported BA | Features | Notes |
|----------|-------------|----------|-------|
| P300 mean amplitude only (current baseline) | 0.670 | 128-dim mean 250-500ms, hcue only | Our v2.0 result |
| P300 enriched 640-dim (v2.0 Phase 7) | 0.554 | peak+latency+AUC+N200, 128ch | Regressed -- reverted |
| Multi-condition fusion (v2.0 Phase 9) | 0.521 | 384-dim hcue+fcue+scue | Degraded |
| Parietal ROI 5-ch (Fz,Pz,P3,P4,Oz) | ~0.65-0.75 estimated | Small-sample P300 BCI literature | arXiv 2510.21969 -- MEDIUM confidence |
| Auditory P300 at Pz/Cz for MDD | ~0.70 reported | 2-3 parietal channels | Frontiers Psychiatry 2022 -- MEDIUM confidence |
| SVM-RFE feature selection on EEG depression | 93.54% accuracy | Selected features | MDPI 2024 -- LOW confidence (different task/dataset) |

Key literature findings:
- P300 is maximal at parietal sites (Pz strongest) -- HIGH confidence (Wikipedia P300, clinical consensus)
- Parietal subset {Fz, Pz, P3, P4, Oz} sufficient for cross-dataset P300 classification -- MEDIUM confidence (arXiv 2510.21969, 2024)
- LPP reduction in MDD: medrxiv 2020 preprint found NO evidence for reduced LPP in MDD -- MEDIUM confidence (contradicts earlier claims)
- Permutation importance corrects for correlated features vs raw SVM/LR weights -- HIGH confidence (Bioinformatics 2010, sklearn docs)
- SVM-RFE feature selection improves EEG depression classification -- MEDIUM confidence (MDPI Applied Sciences 2024)

---

## Sources

- [Adaptive Split-MMD for Small-Sample P300 (arXiv 2510.21969, 2024)](https://arxiv.org/html/2510.21969v1) -- MEDIUM confidence
- [Data-Driven ROI Selection for P300 BCIs (arXiv 2511.02735, 2024)](https://arxiv.org/html/2511.02735v1) -- MEDIUM confidence
- [Impact of Feature Selection on EEG Depression Detection (MDPI Applied Sciences, 2024)](https://www.mdpi.com/2076-3417/14/22/10532) -- MEDIUM confidence
- [Lack of Evidence for Reduced LPP in MDD (medrxiv, 2020)](https://www.medrxiv.org/content/10.1101/2020.04.29.20085571v2.full.pdf+html) -- MEDIUM confidence
- [Permutation importance: a corrected feature importance measure (Bioinformatics, 2010)](https://academic.oup.com/bioinformatics/article/26/10/1340/193348) -- HIGH confidence
- [EEG-based MDD recognition by neural oscillation and asymmetry (Frontiers Neuroscience, 2024)](https://www.frontiersin.org/articles/10.3389/fnins.2024.1362111/full) -- MEDIUM confidence
- [SHAP-based interpretable EEG framework for depression (J Neuroeng Rehab, 2025)](https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-025-01645-5) -- MEDIUM confidence
- [P300 (neuroscience) -- Wikipedia](https://en.wikipedia.org/wiki/P300_(neuroscience)) -- HIGH confidence (consensus summary)

---
*Feature research for: ERP task-state MDD vs HC classification (MODMA dot-probe, 128-channel) -- v3.0 ERP Deep Analysis*
*Researched: 2026-02-23*
