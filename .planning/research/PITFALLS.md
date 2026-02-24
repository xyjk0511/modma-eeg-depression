# Pitfalls Research

**Domain:** ERP task-state MDD classification — N200/LPP components, electrode selection, time-window analysis, feature importance (MODMA dot-probe, N=52)
**Researched:** 2026-02-23
**Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: Feature Enrichment Increases Dimensionality Faster Than It Adds Signal (Confirmed in Phase 7)

**What goes wrong:**
Adding more ERP features per channel multiplies dimensionality without multiplying discriminative information. Phase 7 demonstrated this exactly: expanding from 128-dim (P300 mean) to 640-dim (P300 mean+peak+latency+AUC + N200 mean) regressed BA from 0.670 to 0.554. The pipeline reverted to 128-dim to recover the baseline.

**Why it happens:**
With N=52 subjects and LOSO (51 training samples per fold), the feature-to-sample ratio is the binding constraint. 640 dims / 51 training samples ≈ 12.5 — far above the safe threshold of ~0.2. PCA(20) inside the pipeline cannot fully compensate because the 640 features are correlated within-channel, so PCA components capture within-channel variance rather than between-subject discriminative variance.

**How to avoid:**
- Before adding any new feature type, compute the projected feature-to-sample ratio: new_dims / (N-1). If > 2, reduce first.
- For electrode selection: reduce to a small channel subset (5-10 channels from Pz/Cz/Fz) before adding feature types, not after.
- For LPP (500-800ms): add as a single scalar per electrode, not per-channel mean across all 128 channels, unless electrode selection is applied first.
- Target total features < N_train / 5 = 51/5 ≈ 10 after any reduction step.

**Warning signs:**
BA drops below 0.670 baseline after adding features. Feature matrix shape exceeds (52, 200). PCA(20) retains < 50% variance.

**Phase to address:** Electrode selection phase (must precede any feature type expansion)

---

### Pitfall 2: Electrode Selection by Full-Dataset Topographic Difference Map Is Leakage

**What goes wrong:**
Selecting the top-K electrodes by MDD vs HC amplitude difference computed on all 52 subjects, then using those electrodes as features in LOSO, is feature selection leakage. The test subject's data influenced which electrodes were selected.

**Why it happens:**
Topographic maps are a standard ERP visualization tool. Researchers naturally inspect the grand-average difference map and pick the channels with the largest effect. This feels like domain knowledge but is data-driven selection on the full dataset.

**How to avoid:**
- Use literature-defined channel sets: P300 → {Pz, Cz, P3, P4, CPz}; N200 → {Fz, FCz, Cz}; LPP → {Pz, P3, P4, POz}.
- Or perform electrode selection inside the LOSO loop: for each fold, select top-K channels on training subjects only, apply the same K channels to the test subject.
- Never select channels by inspecting the full-dataset MDD-HC difference map and then using those channels in the same LOSO evaluation.

**Warning signs:**
Channel selection step occurs before loo.split(). Selected channels are the same across all LOSO folds. BA is suspiciously high (>0.80) after channel selection.

**Phase to address:** Electrode selection phase — must be inside LOSO loop or use literature-defined sets

---

### Pitfall 3: Time-Window t-test on Full Dataset Then Using That Window in LOSO Is Double-Dipping

**What goes wrong:**
Running a group t-test across all 52 subjects to find the most discriminative 50ms time bin, then using that bin as the feature window in LOSO, is circular. The test subject's data determined the window.

**Why it happens:**
Time-window analysis is described as exploratory or descriptive but the window is then used in the same classification pipeline. The distinction between exploration and exploitation is lost.

**How to avoid:**
- Time-window t-tests are valid as a standalone descriptive analysis (report which windows are significant, do not feed back into LOSO).
- If the goal is to find the optimal window for classification, do it inside the LOSO loop: for each fold, find the best window on training subjects, apply to test subject.
- Safest approach: use literature-defined windows (P300: 250-500ms, N200: 100-250ms, LPP: 500-800ms) for classification; report t-test results separately as dataset characterization.

**Warning signs:**
The time window used in load_erp_features() was chosen based on a t-test run on the full dataset. The same subjects appear in both the t-test and the LOSO evaluation.

**Phase to address:** Time-window analysis phase — keep descriptive analysis and classification pipeline strictly separate

---

### Pitfall 4: Feature Importance from Classifier Weights Interpreted as Channel Importance Without Accounting for PCA Rotation

**What goes wrong:**
The current pipeline is StandardScaler → PCA(20) → LogisticRegression. The classifier weights are in PCA component space, not in original channel space. Mapping weights back to channels requires multiplying by the PCA loading matrix. Skipping this step produces meaningless channel importance values.

**Why it happens:**
clf.coef_ is readily accessible. Researchers report these directly as feature weights without accounting for the PCA transformation.

**How to avoid:**
- To get channel-space weights: channel_weights = pca.components_.T @ clf.coef_.flatten() (shape: n_channels).
- Or use permutation importance directly on the original feature matrix (before PCA), which is model-agnostic and does not require weight back-projection.
- If using permutation importance, permute one feature at a time and measure BA drop.
- Report which approach was used and its limitations.

**Warning signs:**
Feature importance reported as clf.coef_ shape (1, 20) — these are PCA component weights, not channel weights. Channel importance map has 20 values instead of 128.

**Phase to address:** Feature importance phase — verify weight back-projection or use permutation importance

---

### Pitfall 5: LPP Window (500-800ms) Extends Beyond Epoch End

**What goes wrong:**
If the epoch end is at 700ms or 800ms, the LPP window (500-800ms) either gets truncated or captures the epoch boundary artifact. Mean amplitude in a truncated window is not comparable across subjects with different epoch lengths.

**Why it happens:**
LPP literature uses 500-800ms or even 500-1000ms. Researchers apply these windows without checking the actual epoch duration in the dataset.

**How to avoid:**
- Before defining LPP_WIN, check epochs.tmax in the MODMA data. If tmax < 0.8, truncate LPP_WIN to (0.5, tmax - 0.05) to avoid boundary effects.
- Print times[-1] from the loaded epoch data and assert it is >= LPP_WIN[1] before extracting features.
- If epoch duration is insufficient for LPP, report this as a dataset limitation and skip LPP.

**Warning signs:**
lpp_mask returns fewer time points than expected. times[-1] < 0.8. Mean LPP amplitude is near zero for all subjects.

**Phase to address:** LPP feature extraction phase — verify epoch duration before defining window

---
### Pitfall 6: ERP Averaging Leakage (Pre-existing)

**What goes wrong:**
Fitting StandardScaler/PCA on all subjects before the LOSO loop constitutes leakage.

**How to avoid:**
- StandardScaler and PCA must be fit inside the LOSO loop on training subjects only.
- Fixed time windows from literature are safe.

**Warning signs:**
BA > 0.80 on LOSO with N=52. Any fit or transform call outside the Pipeline object.

**Phase to address:** All feature extraction phases

---

### Pitfall 7: Multiple Comparisons Across Conditions (Pre-existing)

**What goes wrong:**
Running permutation tests for hcue, fcue, scue then reporting the best p-value inflates Type I error by 3x.

**How to avoid:**
- Apply Bonferroni correction (p<0.017 per condition) or pre-specify the primary condition (hcue).
- Report all condition results regardless of significance.

**Phase to address:** Statistical validation phase

---## Moderate Pitfalls

### Pitfall 8: N200/P300 Correlation Adds Dimensionality Without Information (Confirmed Phase 7)

**What goes wrong:**
Phase 7 showed adding N200 mean (128-dim) to P300 features degraded BA from 0.670 to 0.554.
The N200 window likely captures the rising edge of the P300 rather than a distinct trough.

**How to avoid:**
- Verify a distinct negative deflection exists in 100-250ms in the grand-average ERP first.
- Check correlation between N200 mean and P300 mean: if |r| > 0.5, they are redundant.
- If N200 is added, use mean across 5 frontal channels only, not all 128.

**Phase to address:** N200 feature extraction phase

---

### Pitfall 9: Permutation Importance on Full Dataset Leaks Test Subject Information

**What goes wrong:**
Permutation importance computed on all 52 subjects uses the test subject to estimate importance.
Valid for descriptive analysis but not for selecting features to use in LOSO.

**How to avoid:**
- Descriptive use: permutation importance on full dataset is acceptable, labeled as descriptive.
- Feature selection use: compute inside each LOSO fold on training subjects only.

**Phase to address:** Feature importance phase

---

### Pitfall 10: Latency Features Are Noisy at N=52 (Confirmed Phase 7)

**What goes wrong:**
Phase 7 included P300 latency per channel (128-dim) and BA regressed to 0.554.
Peak latency (argmax) is a single-sample measurement with high variance at N=52.

**How to avoid:**
- Treat latency as secondary to amplitude features.
- If included, use centroid of the P300 window rather than argmax.
- Run ablation: if BA drops when latency is added, exclude it.

**Phase to address:** Feature enrichment phases

---

### Pitfall 11: Multi-Condition Fusion Degrades Performance (Confirmed Phase 9)

**What goes wrong:**
Phase 9: 384-dim fusion BA=0.521, p=0.412 — worse than hcue alone (BA=0.670, p=0.022).
Adding conditions introduces noise that overwhelms the hcue signal.

**How to avoid:**
- Treat multi-condition fusion as ablation, not primary analysis.
- If fusion is attempted, reduce each condition to 3 or fewer PCA components first.

**Phase to address:** Any future multi-condition analysis

---## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| All 128 channels | No channel selection | PCA uninterpretable for importance | OK with PCA inside CV; not if importance is goal |
| Fixed P300 window | No tuning needed | May miss actual peak | OK if grand-average ERP verified |
| Single condition (hcue) | Avoids dimensionality problem | Misses condition effects | OK as primary result (confirmed Phase 9) |
| Mean amplitude only | Low-variance feature | Misses latency | OK — Phase 7 showed latency+AUC hurt |
| Literature electrode sets | No leakage risk | May not be optimal | OK — safer than data-driven selection |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PCA + feature importance | Report clf.coef_ as channel weights | Back-project: pca.components_.T @ clf.coef_.flatten() |
| Time-window t-test + LOSO | Use t-test window in same LOSO | Keep t-test descriptive; use literature windows in LOSO |
| Electrode selection + LOSO | Select on full dataset, use in LOSO | Select inside LOSO loop or use literature channels |
| LPP window + epoch duration | Apply 500-800ms without checking tmax | Assert times[-1] >= LPP_WIN[1] before extraction |
| N200 + P300 features | Concatenate all 128 channels for both | Verify distinct N200 trough; reduce to frontal channels |

---## Looks Done But Isnt Checklist

- [ ] **Electrode selection:** Literature-defined or inside LOSO loop, not on full dataset
- [ ] **Time-window t-test:** Reported separately from LOSO, not used to define LOSO feature window
- [ ] **Feature importance:** Weights back-projected through PCA, or use permutation importance
- [ ] **LPP window:** times[-1] >= 0.8 before defining LPP_WIN = (0.5, 0.8)
- [ ] **N200 grand-average:** Distinct negative deflection in 100-250ms verified before adding N200
- [ ] **Dimensionality:** Total features < 10 after reduction (N=52 constraint)
- [ ] **Scaler/PCA inside CV:** Inside Pipeline object, not fit before loo.split()
- [ ] **Permutation level:** len(y) == n_subjects, not n_trials
- [ ] **Feature enrichment ablation:** BA does not drop below 0.670 baseline after adding features

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Feature enrichment regresses BA (Phase 7) | LOW | Revert to 128-dim p300_mean; add features one type at a time with ablation |
| Electrode selection leakage | MEDIUM | Re-run with literature channels or inside LOSO; re-run permutation test |
| Time-window double-dipping | MEDIUM | Separate t-test from LOSO; re-run LOSO with literature windows |
| PCA weight back-projection missing | LOW | Add pca.components_.T @ clf.coef_.flatten() post-hoc |
| LPP window truncated | LOW | Check tmax; adjust LPP_WIN; re-extract features |
| Scaler fit outside CV | LOW | Move inside Pipeline; re-run LOSO |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Feature enrichment dimensionality (P1) | Electrode selection phase (before feature expansion) | Assert features.shape[1] < 10 after reduction |
| Electrode selection leakage (P2) | Electrode selection phase | Literature-defined or inside LOSO loop |
| Time-window double-dipping (P3) | Time-window analysis phase | t-test and LOSO use independent windows |
| Feature importance PCA back-projection (P4) | Feature importance phase | Weight shape is (128,) not (20,) |
| LPP epoch duration (P5) | LPP feature extraction phase | Assert times[-1] >= LPP_WIN[1] |
| ERP averaging leakage (P6) | All feature extraction phases | No cross-subject ops outside Pipeline |
| Multiple comparison inflation (P7) | Statistical validation phase | Bonferroni or pre-specified primary condition |
| N200/P300 correlation (P8) | N200 feature extraction phase | Grand-average ERP verified; feature correlation checked |
| Permutation importance leakage (P9) | Feature importance phase | Descriptive vs. selection use distinguished |
| Latency feature noise (P10) | Feature enrichment phases | Ablation: BA does not drop when latency added |
| Multi-condition fusion degradation (P11) | Any future fusion phase | Pre-reduce to 3 or fewer components per condition |

---

## Sources

- Phase 7 empirical result: 640-dim regressed BA 0.670 to 0.554 — .planning/phases/07-p300-feature-enrichment-n200/07-01-SUMMARY.md. HIGH confidence.
- Phase 8 empirical result: 128-dim P300 mean BA=0.670, p=0.022 — .planning/phases/08-permutation-test-validation/08-01-SUMMARY.md. HIGH confidence.
- Phase 9 empirical result: 384-dim fusion BA=0.521, p=0.412 — .planning/phases/09-multi-condition-fusion-contrast-features/09-VERIFICATION.md. HIGH confidence.
- Brookshire et al., Frontiers in Neuroscience 2024 — data leakage in translational EEG. HIGH confidence.
- Saarschmidt et al., Scientific Reports 2021 — inflated accuracy from feature selection leakage. HIGH confidence.
- Nature Mental Health 2023 — EEG biomarkers for MDD. HIGH confidence.
- Codebase analysis: run_modma_erp.py — current ERP pipeline. HIGH confidence (direct inspection).

---
*Pitfalls research for: ERP deep analysis — N200/LPP components, electrode selection, time-window analysis, feature importance (v3.0 milestone)*
*Researched: 2026-02-23*