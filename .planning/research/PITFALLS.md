# Domain Pitfalls

**Domain:** ERP task-state MDD classification -- P300/N200 features, multi-condition fusion (MODMA dot-probe)
**Researched:** 2026-02-22
**Confidence:** HIGH

---

## Critical Pitfalls
### Pitfall 1: ERP Averaging Computed on Full Dataset Before LOSO Split

**What goes wrong:**
The average ERP per subject is computed per-subject independently -- this is safe. The risk is future additions: computing a grand-average ERP across all subjects to define a peak-detection window, or fitting StandardScaler/PCA on all subjects before the LOSO loop. Both constitute leakage.

**Why it happens:**
ERP analysis naturally involves group-level operations (grand averages, component latency estimation). Researchers apply these globally for convenience.

**Consequences:**
Inflated BA. Saarschmidt et al. (Scientific Reports 2021) showed feature selection outside CV inflates accuracy by 10-20% in neuropsychiatric biomarker studies.

**How to avoid:**
- StandardScaler must be fit inside the LOSO loop on training subjects only -- current Pipeline does this correctly, verify it stays
- PCA must be fit inside the LOSO loop -- current PCA(n_components=20) inside Pipeline is correct
- Fixed time windows (250-500ms for P300, 100-250ms for N200) are safe -- they come from literature, not data
- Any peak-detection window derived from data must not use the test subject

**Warning signs:**
BA suspiciously high (>0.80) on LOSO with N=52. Any fit or transform call outside the Pipeline object.

**Phase to address:** ERP feature extraction phase

---
### Pitfall 2: Trial Rejection Threshold Applied Globally Creates Subject-Level Leakage

**What goes wrong:**
The current code uses reject=dict(eeg=150e-6) as a fixed amplitude threshold -- this is safe. The risk is if a data-adaptive threshold (e.g., median + k*MAD across all subjects) is used to set the rejection criterion. That would use test-subject trial statistics to define the threshold.

**Why it happens:**
Adaptive thresholds are better practice for artifact rejection in general EEG, but in a LOSO loop they must be computed per-training-fold or be fixed a priori.

**Consequences:**
The set of included subjects changes depending on which subject is held out, making the evaluation non-reproducible and potentially biased.

**How to avoid:**
- Keep reject=dict(eeg=150e-6) as a fixed threshold (current approach is correct)
- The data.shape[0] < 10 minimum trial count is a fixed threshold -- safe
- If adaptive thresholds are added, compute them only on training subjects within each LOSO fold

**Warning signs:**
Subject inclusion set changes when you change the random seed or fold order.

**Phase to address:** ERP feature extraction phase

---
### Pitfall 3: Multi-Condition Feature Concatenation Triples Dimensionality Without Tripling Information

**What goes wrong:**
Concatenating hcue + fcue + scue features produces 3x the feature dimensions (e.g., 128 channels x 3 conditions = 384 dims) for the same N=52 subjects. With LOSO, each fold trains on 51 subjects. The effective degrees of freedom are ~51 subjects -- the conditions are correlated within-subject. A classifier with 384 features and 51 training samples will overfit severely.

**Why it happens:**
Multi-condition fusion seems like "more data" but it is more features from the same subjects. The feature-to-sample ratio worsens, not improves.

**Consequences:**
BA near chance or worse than single-condition. Current results show this pattern: hcue BA=0.670, fcue BA=0.410, scue BA=0.536. Naive concatenation will likely produce BA near 0.5 unless dimensionality is aggressively reduced.

**How to avoid:**
- Apply PCA inside the LOSO loop before concatenation, or reduce each condition to a small number of components first
- Target total features < N_train / 5 = 51/5 ~ 10 features after fusion
- Consider condition-level aggregation: compute a single scalar per condition (e.g., mean frontal P300 amplitude) rather than all 128 channels
- Alternatively, use only the best single condition (hcue) and treat multi-condition as an ablation

**Warning signs:**
Feature matrix shape after concatenation exceeds 50 columns. BA drops below single-condition baseline after fusion.

**Phase to address:** Multi-condition fusion phase

---
### Pitfall 4: P300/N200 Window Fixed at Literature Values May Miss Actual Component in This Dataset

**What goes wrong:**
The P300 window (250-500ms) is standard for auditory/visual oddball paradigms. The MODMA dot-probe task uses a different stimulus type and timing. The actual P300 peak may be shifted (e.g., 300-600ms for cognitive tasks). Using a fixed window that misses the actual peak produces near-zero discriminative features.

**Why it happens:**
Researchers apply standard windows from the literature without verifying them on the actual dataset grand-average ERP.

**Consequences:**
Features capture noise rather than the component of interest. BA near chance even if the component genuinely differs between MDD and HC.

**How to avoid:**
- Plot the grand-average ERP (mean across all subjects) for each condition before defining feature windows
- Verify a positive deflection exists in the 250-500ms range for hcue condition
- If the peak is shifted, adjust the window based on the grand average -- this is a fixed, data-independent decision made before any CV, not leakage
- For N200: verify a negative deflection exists in 100-250ms

**Warning signs:**
Mean P300 amplitude across all subjects is near zero or negative. Feature variance is very low.

**Phase to address:** ERP feature extraction phase (verification step before LOSO)

---
### Pitfall 5: Multiple Comparisons Across Conditions Inflates Type I Error

**What goes wrong:**
Running permutation tests separately for hcue, fcue, and scue, then reporting the best p-value, inflates Type I error by a factor of 3. With 3 conditions and alpha=0.05, the expected false positive rate is ~14%.

**Why it happens:**
Each condition is analyzed independently and the best result is highlighted. This is implicit p-hacking even without intent.

**Consequences:**
A p<0.05 result for one condition may be a false positive. The current hcue BA=0.670 needs a valid permutation p-value with multiple comparison correction.

**How to avoid:**
- Apply Bonferroni correction: require p<0.017 per condition if testing 3 conditions
- Or pre-specify the primary condition (hcue, as it has the highest BA) before running permutation tests
- Report all three condition results regardless of significance

**Warning signs:**
Only the best-performing condition has a permutation test run. p-values reported without multiple comparison correction.

**Phase to address:** Statistical validation phase

---
### Pitfall 6: Permutation Test Permutes at Wrong Level

**What goes wrong:**
In the ERP pipeline, each subject contributes exactly one feature vector (the avg-ERP). LOSO operates at subject level. If a permutation test permutes the y array at the trial level rather than the subject level, the null distribution is invalid -- too narrow, producing falsely significant p-values.

**Why it happens:**
Copy-paste from the resting-state pipeline where windows were the unit of analysis.

**Consequences:**
Null distribution is too narrow, producing falsely significant p-values.

**How to avoid:**
- In the ERP pipeline, y has one entry per subject -- permuting y directly is correct
- Assert len(y) == len(np.unique(subject_ids)) before running permutation test
- If trial-level features are added later, permute subject labels and propagate to all trials of that subject

**Warning signs:**
len(y) equals number of trials rather than number of subjects.

**Phase to address:** Statistical validation phase

---

## Moderate Pitfalls

### Pitfall 7: N200 and P300 Features Are Highly Correlated

**What goes wrong:**
N200 (100-250ms) and P300 (250-500ms) are adjacent in time. If the P300 is large, the N200 window may capture the rising edge of the P300 rather than the N200 trough. The two features will be highly correlated, adding dimensionality without adding information.

**How to avoid:**
- Verify N200 and P300 features have low correlation (|r| < 0.5) across subjects
- If correlation is high, use only the more discriminative feature
- Consider peak amplitude (min in N200 window, max in P300 window) rather than mean amplitude to better isolate each component

**Phase to address:** ERP feature extraction phase

---
### Pitfall 8: Latency Features Are Noisy at N=52

**What goes wrong:**
Peak latency (argmax of avg-ERP in the P300 window) is a single-sample measurement with high variance. With only 52 subjects and noisy avg-ERPs, latency features add noise rather than signal.

**How to avoid:**
- Treat latency features as secondary to amplitude features
- If latency is included, use a smoothed estimate (centroid of the P300 window rather than argmax)
- Evaluate latency features F-scores in ablation -- if F < 2.0, exclude

**Phase to address:** ERP feature extraction phase

---

### Pitfall 9: Channel Selection by Inspecting Full-Dataset Topographic Map

**What goes wrong:**
If channels are selected based on which channels show the largest MDD vs HC difference in the full dataset, and those channels are then used as features in LOSO, this is feature selection leakage. The current approach uses all 128 channels + PCA inside Pipeline -- this is safe.

**How to avoid:**
- Use literature-defined channel sets (Pz, Cz, Fz for P300) rather than data-driven selection
- Or use PCA inside the LOSO loop (current approach -- keep it)
- Never select channels by inspecting the full-dataset difference map

**Phase to address:** ERP feature extraction phase

---
## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| All 128 channels as features | No channel selection needed | 128 dims forces PCA; PCA components uninterpretable | Acceptable with PCA inside CV |
| Fixed P300 window (250-500ms) | No dataset-specific tuning | May miss actual peak if shifted | Acceptable if grand-average ERP verified first |
| Single condition (hcue only) | Avoids multi-condition dimensionality problem | Misses condition-specific MDD effects | Acceptable as baseline |
| Mean amplitude only (no latency, no area) | Simple, low-variance feature | Misses latency differences | Acceptable for initial validation |

---

## "Looks Done But Isn't" Checklist

- [ ] **ERP averaging:** Verify avg_erp is computed per-subject independently, not using any cross-subject statistics
- [ ] **Scaler/PCA inside CV:** Verify StandardScaler and PCA are inside the Pipeline object, not fit before loo.split()
- [ ] **Grand-average ERP plotted:** Verify P300 peak exists in 250-500ms window before reporting features
- [ ] **Permutation test at subject level:** Verify len(y) == n_subjects, not n_trials
- [ ] **Multiple comparison correction:** If testing 3 conditions, apply Bonferroni or pre-specify primary condition
- [ ] **Multi-condition dimensionality:** If concatenating conditions, verify total features < 10 after reduction
- [ ] **Trial count balance:** Verify MDD and HC subjects have similar trial counts (unequal trials -> unequal SNR -> confound)

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Scaler fit outside CV | LOW | Move scaler inside Pipeline; re-run LOSO |
| Wrong P300 window | LOW | Plot grand-average ERP; adjust window; re-run |
| Multi-condition dimensionality explosion | MEDIUM | Add PCA(n_components=5) per condition inside CV; re-run |
| Multiple comparison inflation | LOW | Apply Bonferroni; re-report p-values |
| Permutation at trial level | MEDIUM | Rewrite permutation loop to permute subject labels; re-run 1000 permutations |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| ERP averaging leakage (Pitfall 1) | ERP feature extraction | Assert no cross-subject ops outside Pipeline |
| Trial rejection threshold leakage (Pitfall 2) | ERP feature extraction | Confirm reject dict is fixed, not data-derived |
| Multi-condition dimensionality (Pitfall 3) | Multi-condition fusion | Assert features.shape[1] < 15 after fusion |
| Wrong P300/N200 window (Pitfall 4) | ERP feature extraction | Plot grand-average ERP before LOSO |
| Multiple comparison inflation (Pitfall 5) | Statistical validation | Bonferroni or pre-specified primary condition |
| Permutation at wrong level (Pitfall 6) | Statistical validation | Assert len(y) == n_subjects |
| N200/P300 correlation (Pitfall 7) | ERP feature extraction | Check feature correlation matrix |
| Latency feature noise (Pitfall 8) | ERP feature extraction | F-score ablation; exclude if F < 2.0 |
| Channel selection leakage (Pitfall 9) | ERP feature extraction | Use literature channels or PCA inside CV only |

---

## Sources

- [Data leakage in deep learning studies of translational EEG](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1373515/full) -- Brookshire et al., Frontiers in Neuroscience 2024. HIGH confidence.
- [Inflated prediction accuracy of neuropsychiatric biomarkers caused by data leakage in feature selection](https://www.nature.com/articles/s41598-021-87157-3) -- Saarschmidt et al., Scientific Reports 2021. HIGH confidence.
- [Risk of data leakage in estimating diagnostic performance for psychiatric disorders](https://www.nature.com/articles/s41598-023-43542-8) -- Nature Scientific Reports 2023. HIGH confidence.
- [Striking a balance: analyzing unbalanced ERP data](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00555/full) -- Frontiers in Psychology 2015. MEDIUM confidence.
- [Technical and clinical considerations for EEG-based biomarkers for MDD](https://www.nature.com/articles/s44184-023-00038-7) -- Nature Mental Health 2023. HIGH confidence.
- Codebase analysis: run_modma_erp.py -- current ERP pipeline with LOSO, avg-ERP per subject, 128-channel P300 features. HIGH confidence (direct inspection).

---
*Pitfalls research for: ERP task-state MDD classification (P300/N200 features, multi-condition fusion)*
*Researched: 2026-02-22*
