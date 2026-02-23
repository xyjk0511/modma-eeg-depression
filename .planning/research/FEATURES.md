# Feature Research

**Domain:** ERP task-state MDD vs HC binary classification (dot-probe hcue/fcue/scue, 128-channel, MODMA)
**Researched:** 2026-02-22
**Confidence:** MEDIUM (multiple web sources + official literature)

## Context: What Already Exists

Existing pipeline (resting-state): PSD band power, FAA, Riemannian covariance, theta/beta ratio, MNE epoch extraction.
Current ERP baseline: hcue avg-ERP P300 mean amplitude per channel -> BA=0.670 (52 subjects, LOSO).
This file covers ONLY new ERP task-state features to add on top of that baseline.

---

## Feature Landscape

### Table Stakes (Expected in Any ERP-Based MDD Pipeline)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| P300 peak amplitude (per channel) | Most replicated ERP biomarker for MDD. MDD shows reduced P300 amplitude vs HC. Peak is more precise than mean. | LOW | avg_erp[:, p300_mask].max(axis=1) -- 128 dims. |
| P300 peak latency (per channel) | MDD shows longer P300 latency. Frontal P300 latency is candidate biomarker of MDD (Frontiers 2022). Pz and C3 latency discriminates MDD subtypes. | LOW | times[p300_mask][argmax] -- 128 dims. |
| P300 area under curve (per channel) | Integrates amplitude over time window. More robust than peak to noise. Standard in clinical ERP literature. | LOW | avg_erp[:, p300_mask].sum(axis=1) * dt -- 128 dims. |
| N200 mean amplitude (100-250ms, per channel) | N200 reflects early attentional engagement. MDD shows altered N200 for emotional stimuli. Negative component; MDD shows reduced negativity for happy, enhanced for sad. | LOW | Same extraction as P300 but N200_WIN=(0.10, 0.25). 128 dims. |
| N200 peak amplitude (per channel) | Peak captures the trough of N200. More sensitive than mean for detecting attentional bias differences. | LOW | avg_erp[:, n200_mask].min(axis=1) -- most negative point. 128 dims. |
| Frontal/parietal ROI reduction | Full 128-channel vectors are high-dimensional for N=52. Literature identifies Fz, Cz, Pz as most discriminative for P300; frontal sites for N200. | LOW | Average over ROI channel groups. Reduces 128 dims to ~5-10. |

### Differentiators (Features That Can Push BA Above 0.70)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Multi-condition concatenation (hcue + fcue + scue) | Each condition captures different emotional processing: sad cues show enhanced P3 in MDD (negative bias), happy cues show reduced P1/P3 (positive blunting). ACM 2024: best accuracy 79.84% when all ERP components used as combined feature set. | MEDIUM | Compute features per condition, concatenate vectors. Requires >=10 trials per condition per subject. |
| Condition contrast (scue - hcue) | MDD is hypersensitive to sad, hyposensitive to happy. Difference between sad and happy condition responses is a direct biomarker of asymmetric emotional bias. More informative than either condition alone. | LOW | feat_scue - feat_hcue per feature type. ~128 dims. Requires multi-condition. |
| P300 topographic gradient (frontal/parietal ratio) | MDD shows altered fronto-parietal P300 distribution. Ratio of frontal to parietal P300 amplitude captures this spatial shift. | LOW | mean(frontal_ch) / mean(parietal_ch) per condition. ~3 dims. |
| N200 latency (per channel) | Latency of N200 reflects speed of early attentional capture. MDD shows altered timing of early emotional processing. Complements amplitude. | LOW | times[n200_mask][argmin] -- 128 dims. |
| Single-trial variability (std across trials) | MDD shows increased trial-to-trial variability in ERP responses, reflecting unstable attentional engagement. Mean ERP discards this signal. | MEDIUM | data.std(axis=0)[:, p300_mask].mean(axis=1). Requires keeping raw trial data in memory. |

### Anti-Features (Commonly Considered, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full 128-channel raw ERP vectors without ROI reduction | Maximum information retention | 128ch x 3 features x 3 conditions = 1152 dims for N=52. Extreme curse of dimensionality. | ROI-averaged features + PCA(20). |
| Per-trial classification | More training samples | Trials from same subject are correlated. Trial-level accuracy is inflated. Same problem as window-level for resting-state. | Subject-level LOSO with avg-ERP features. |
| Condition averaging across hcue/fcue/scue | Simplifies feature space | Destroys condition-specific emotional processing differences that are the main signal. | Concatenate per-condition features or compute contrasts. |
| Deep learning on ERP waveforms | Papers report high accuracy | N=52 is far too small for cross-subject generalization. | LR/SVM with PCA on handcrafted ERP features. |

---

## Feature Dependencies



### Dependency Notes

- N200 and P300 share the same epoch data -- extract both in one pass to avoid reloading raw files.
- Multi-condition concatenation requires all 3 conditions to have enough trials. Subjects with <10 trials in any condition must be excluded (may reduce N slightly).
- Single-trial variability conflicts with current memory management (data deleted after avg_erp). Requires architectural change.
- ROI reduction should happen before PCA -- it encodes domain knowledge about which channels matter.

---

## MVP Definition

### Launch With (v1) -- Enrich P300, Add N200, Multi-Condition

- [ ] P300 peak amplitude per channel (LOW)
- [ ] P300 peak latency per channel (LOW)
- [ ] P300 area under curve per channel (LOW)
- [ ] N200 mean + peak amplitude per channel 100-250ms (LOW)
- [ ] Multi-condition concatenation: hcue + fcue + scue (MEDIUM)
- [ ] Condition contrast scue - hcue: direct negative-bias biomarker (LOW, requires multi-condition)

### Add After Validation (v1.x)

- [ ] ROI-averaged features at Fz/Cz/Pz groups
- [ ] Condition contrast fcue - hcue
- [ ] P300 topographic gradient frontal/parietal ratio
- [ ] N200 latency per channel

### Future Consideration (v2+)

- [ ] Single-trial variability -- requires memory architecture change
- [ ] Fusion with resting-state features

---

## Feature Prioritization Matrix

| Feature | Classification Value | Implementation Cost | Priority |
|---------|---------------------|---------------------|----------|
| P300 peak amplitude per channel | HIGH | LOW | P1 |
| P300 peak latency per channel | HIGH | LOW | P1 |
| N200 mean amplitude per channel | HIGH | LOW | P1 |
| P300 area under curve | MEDIUM | LOW | P1 |
| Multi-condition concatenation | HIGH | MEDIUM | P1 |
| Condition contrast scue - hcue | HIGH | LOW | P1 |
| N200 peak amplitude | MEDIUM | LOW | P2 |
| N200 latency | MEDIUM | LOW | P2 |
| ROI reduction Fz/Cz/Pz groups | MEDIUM | LOW | P2 |
| P300 topographic gradient | MEDIUM | LOW | P2 |
| Single-trial variability | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have -- directly addresses milestone goal
- P2: Should have -- adds spatial/temporal specificity after P1 validated
- P3: Nice to have -- marginal gains, higher complexity

---

## Literature Benchmarks

| Approach | Reported BA | Features | Notes |
|----------|-------------|----------|-------|
| P300 mean amplitude only (current) | 0.670 | 128-dim mean 250-500ms, hcue only | Our baseline |
| P300 amplitude + latency + area | ~0.70-0.75 estimated | 3 features x key channels | Standard clinical ERP |
| Multi-condition ERP fusion | 79.84% accuracy | All ERP components, all conditions | ACM 2024 |
| P300 + N200 combined | ~0.72-0.78 estimated | Both components, frontal/parietal | Multiple MDD ERP studies |
| Realistic cross-subject ERP | 0.65-0.80 | Varies | With proper LOSO evaluation |

**Key literature findings:**
- MDD shows reduced P300 amplitude and longer P300 latency vs HC -- HIGH confidence (replicated across multiple studies including iSPOT-D n>1000)
- Frontal P300 latency is candidate biomarker of MDD -- MEDIUM confidence (Frontiers Human Neuroscience 2022)
- Auditory P300 latency at C3 and Pz discriminates MDD subtypes -- MEDIUM confidence (Frontiers Psychiatry 2022)
- MDD shows enhanced P3 to sad cues and reduced P1/P3 to happy cues in dot-probe tasks -- MEDIUM confidence (Frontiers Human Neuroscience 2020)
- N200 reflects early attentional bias; MDD shows reduced N200 for pleasant stimuli -- LOW confidence (single older study + 2014 replication)
- Best accuracy 79.84% when all ERP components used as combined feature set -- MEDIUM confidence (ACM 2024)

---

## Sources

- [Electrophysiological biomarkers and age in MDD (Frontiers Human Neuroscience, 2022)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1055685/full) -- MEDIUM confidence
- [P300 ERPs in MDD subtypes (Frontiers Psychiatry, 2022)](https://www.frontiersin.org/articles/10.3389/fpsyt.2022.1021365) -- MEDIUM confidence
- [Negative Bias in MDD dot-probe ERP (Frontiers Human Neuroscience, 2020)](https://www.frontiersin.org/articles/10.3389/fnhum.2020.593010/full) -- MEDIUM confidence
- [Decoding Functional Brain Data for Emotion Recognition (ACM, 2024)](https://dl.acm.org/doi/10.1145/3657638) -- MEDIUM confidence
- [Depressive states and frontal-parietal ERPs during emotional stimuli (Nature Sci Rep, 2023)](https://www.nature.com/articles/s41598-023-44368-0) -- MEDIUM confidence
- [Emotional Conflict Processing in MDD: N450 and P300 (Frontiers Human Neuroscience, 2018)](https://www.frontiersin.org/articles/10.3389/fnhum.2018.00214/full) -- MEDIUM confidence
- [N200 component of ERPs in depression (Biological Psychiatry, 1993)](https://www.biologicalpsychiatryjournal.com/article/0006-3223(93)90122-T/fulltext) -- LOW confidence (foundational but old)
- [P300 latency as indicator of severity in MDD (PMC, 2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4866344/) -- MEDIUM confidence

---
*Feature research for: ERP task-state MDD vs HC classification (MODMA dot-probe, 128-channel)*
*Researched: 2026-02-22*
