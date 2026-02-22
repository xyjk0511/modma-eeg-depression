# Domain Pitfalls

**Domain:** EEG-based MDD vs HC classification (MODMA 128-channel resting state)
**Researched:** 2026-02-22
**Overall confidence:** HIGH (domain literature + codebase analysis)

## Critical Pitfalls

Mistakes that cause invalid results or require major rework.

### Pitfall 1: Differential QC Dropout Creating Artificial Class Imbalance

**What goes wrong:** When relaxing QC thresholds (e.g. max_bad_channels from 3 to 15), MDD and HC subjects may have systematically different data quality. If MDD patients had more movement artifacts or worse electrode contact, relaxing QC retains more noisy MDD subjects while HC subjects were already clean. The classifier then learns "noisy = MDD" rather than genuine neural signatures.

**Why it happens:** MDD patients often exhibit psychomotor agitation during recording. The MODMA EDF files have documented physical scaling anomalies (DC offsets like [-5460, -3475] uV) that may affect groups differently.

**Consequences:** Above-chance BA driven entirely by artifact-level differences, not neural biomarkers. Permutation test may still pass because the confound is label-correlated.

**Prevention:**
- After QC, compare retained MDD vs HC counts — target ratio within 1.5:1 of original (24:29)
- Log per-subject bad-channel counts; compare distributions between groups (Wilcoxon test)
- If one group loses >20% more subjects than the other, investigate whether QC captures group quality differences vs neural signal

**Detection:** Check `qc_report.csv` — if one group loses >30% more subjects than the other, this pitfall is active.

**Phase:** QC parameter adjustment (Phase 1)

### Pitfall 2: Data Leakage via Window-Level Splitting

**What goes wrong:** Multiple time windows from the same subject end up in both train and test sets. The classifier learns subject-specific EEG signatures (electrode impedance, skull thickness, individual alpha frequency) rather than MDD vs HC differences. Brookshire et al. (Frontiers in Neuroscience 2024) showed "the majority of translational DNN-EEG studies suffer from data leakage."

**Why it happens:** Default sklearn CV splits at sample level. With 5-6 windows per subject, random splitting almost guarantees leakage.

**Consequences:** Reported BA is meaningless — a model memorizing 40 individuals has zero generalization.

**Prevention:**
- MUST use `StratifiedGroupKFold` with `groups=subject_id` for all CV (already implemented — verify it stays)
- Subject-level metric aggregation: average window predictions per subject before computing BA (already implemented via `y_pred_subject_probs`)
- Never use `StratifiedKFold` or `ShuffleSplit` without group constraints

**Detection:** Assert `train_groups.isdisjoint(test_groups)` in every CV loop. Current code has `leakage_detected` flag — keep it.

**Phase:** All phases (invariant constraint)

### Pitfall 3: Feature Selection / Hyperparameter Tuning Outside CV

**What goes wrong:** Feature selection or QC threshold selection performed on full dataset before cross-validation leaks test-set information. Saarschmidt et al. (Scientific Reports 2021) showed this inflates accuracy by 10-20% in neuropsychiatric biomarker studies.

**Why it happens:** Computationally convenient to select features once, then run CV. Current pipeline correctly nests QC search inside outer CV, but refactoring could break this.

**Consequences:** Optimistic BA that does not replicate on held-out data.

**Prevention:**
- All data-dependent decisions (QC threshold, feature selection, hyperparameters) must stay inside CV loop
- `run_full_model_selection()` correctly nests QC search — do not refactor this out
- If adding new feature selection, wrap it in inner CV

**Detection:** Code review: grep for `fit`, `transform`, `SelectKBest`, or threshold selection touching test indices.

**Phase:** Feature engineering and model selection phases

### Pitfall 4: Filter Edge Effects Contaminating Window 0

**What goes wrong:** The first time window after highpass filtering contains filter transient artifacts — elevated bad-channel counts that are not real neural noise. Including Window 0 inflates QC rejection rates and wastes data, or if kept, injects non-neural variance into features.

**Why it happens:** IIR/FIR highpass filters need a settling period. At 0.5 Hz highpass with 250 Hz sampling, the transient can last 2-6 seconds — a significant portion of a 10-second window. PROJECT.md confirms: "Window 0 bad channel count is far higher than subsequent windows."

**Consequences:** Either excessive subject dropout (if QC catches it) or contaminated features (if QC misses it). Both degrade classification.

**Prevention:**
- Skip Window 0 unconditionally after filtering
- Alternatively, add `raw.crop(tmin=6.0)` before windowing to discard the transient period
- Verify by comparing bad-channel distributions of Window 0 vs Windows 1+ across all subjects

**Detection:** Plot per-window bad-channel counts — Window 0 should not be a consistent outlier after the fix.

**Phase:** QC parameter adjustment (Phase 1)

## Moderate Pitfalls

### Pitfall 5: Overfitting with 144 Features on ~40 Subjects

**What goes wrong:** The current pipeline extracts 144-dimensional features from ~35-40 retained subjects (each contributing 5-6 windows, so ~200 samples). With p >> n at the subject level, classifiers memorize noise. Even with CV, the effective degrees of freedom are limited by the ~40 independent subjects, not the ~200 correlated windows.

**Why it happens:** Feature count grew organically (PSD + DE + Hjorth + asymmetry + connectivity + Riemannian = 144 dims). Each addition seemed small, but the cumulative dimensionality is excessive for this sample size.

**Prevention:**
- Apply dimensionality reduction inside CV: PCA(n_components=0.95) or SelectKBest(k=20-30)
- Rule of thumb: effective features should be < n_subjects / 5, so target ~8 features for 40 subjects
- Consider region-level aggregation to reduce redundancy before classification

**Detection:** Compare CV BA with and without PCA — if PCA improves BA, overfitting was present.

**Phase:** Feature engineering (Phase 2)

### Pitfall 6: Amplitude Threshold Insensitive to EDF Scaling Anomalies

**What goes wrong:** The current `bad_amp_uv=200` threshold assumes data is in standard microvolt range. But PROJECT.md documents that MODMA EDF headers have large DC offsets (physical range like [-5460, -3475] uV). After highpass filtering the DC is removed, but residual scaling issues may leave some channels with amplitudes that are systematically above or below the threshold — rejecting valid data or passing bad data.

**Why it happens:** EDF physical min/max calibration varies across recording sessions. A fixed uV threshold does not adapt to per-file scaling.

**Prevention:**
- After filtering, check actual amplitude distributions per subject before applying threshold
- Consider percentile-based rejection (e.g. reject channels > 99th percentile of all channels) instead of fixed uV
- At minimum, test multiple thresholds (200, 300, 400 uV) — already in `bad_amp_candidates`

**Detection:** Log median and max amplitude per subject after filtering. If values cluster far from 200 uV, the threshold is miscalibrated.

**Phase:** QC parameter adjustment (Phase 1)

### Pitfall 7: Permutation Test Invalidated by Correlated Windows

**What goes wrong:** The permutation test shuffles subject-level labels and reruns the full pipeline, but the p-value assumes independent samples. With 5-6 correlated windows per subject, the effective sample size is ~40, not ~200. If the permutation test uses window-level BA instead of subject-level BA, the null distribution is too narrow, producing falsely significant p-values.

**Why it happens:** The current `build_report()` correctly permutes at the subject level and evaluates subject-level BA. But the permutation rerun uses `run_full_model_selection()` which includes inner CV — if inner CV folds are too small after permutation, some permutations silently fail and get dropped, biasing the null distribution.

**Prevention:**
- Always permute at subject level (already correct)
- Always evaluate subject-level BA in permutations (already correct)
- Log the fraction of failed permutations — if >10% fail, the null distribution is biased
- Use at least 1000 permutations for reliable p-value at alpha=0.05

**Detection:** Check `effective_permutations` in metrics.json — should be >900 out of 1000.

**Phase:** Statistical validation (Phase 3)

### Pitfall 8: Average Reference Amplifying Bad Channels

**What goes wrong:** The pipeline applies average reference (`raw.set_eeg_reference('average')`) before QC. If a few channels have extreme amplitudes (common with MODMA's scaling issues), the average reference spreads that noise to ALL channels, making previously good channels appear bad.

**Why it happens:** Average reference subtracts the mean across channels at each time point. One channel at 500 uV shifts all 127 other channels by ~4 uV — small individually, but systematic.

**Prevention:**
- Identify and interpolate or exclude grossly bad channels BEFORE applying average reference
- Or use robust average reference (median instead of mean)
- At minimum, run QC twice: once to find extreme channels, exclude them from reference, then re-reference

**Detection:** Compare bad-channel counts before vs after re-referencing. If re-referencing increases bad channels, this pitfall is active.

**Phase:** Preprocessing (Phase 1)

## Minor Pitfalls

### Pitfall 9: Connectivity Features Dominated by Volume Conduction

**What goes wrong:** Region-averaged signals used for connectivity (ImCoh) reduce spatial resolution. Averaging channels within a region creates a "virtual electrode" that may capture volume-conducted activity rather than true inter-regional communication. The imaginary part of coherence mitigates zero-lag volume conduction, but region-averaging can introduce artificial phase delays.

**Prevention:**
- Use representative single channels per region instead of averages for connectivity
- Verify ImCoh values are in plausible range (0.01-0.3 for resting state)
- Compare connectivity features' discriminative power (f_classif scores) against simpler PSD features

**Phase:** Feature engineering (Phase 2)

### Pitfall 10: Medication Confound in MDD Group

**What goes wrong:** MDD patients in MODMA are likely on psychotropic medication (SSRIs, benzodiazepines) which directly alter EEG power spectra — particularly alpha and beta bands. The classifier may learn medication signatures rather than depression biomarkers.

**Prevention:**
- Check MODMA participants.tsv for medication status columns
- If medication data available, include as covariate or run sensitivity analysis excluding medicated subjects
- Report this as a known limitation if medication data is unavailable

**Phase:** Interpretation (Phase 3)

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| QC parameter relaxation | Pitfall 1: Differential dropout | Compare MDD/HC retention rates in qc_report.csv |
| QC parameter relaxation | Pitfall 6: Amplitude miscalibration | Log amplitude distributions, test multiple thresholds |
| Skip Window 0 | Pitfall 4: Filter transient | Verify bad-channel counts normalize after W0 removal |
| Preprocessing | Pitfall 8: Average reference noise spread | Exclude bad channels before re-referencing |
| Feature engineering | Pitfall 5: Overfitting (144 dims, ~40 subjects) | PCA or SelectKBest inside CV, target <10 effective features |
| Feature engineering | Pitfall 9: Volume conduction in connectivity | Check ImCoh plausibility, compare vs PSD-only baseline |
| Model selection | Pitfall 3: Tuning outside CV | All data-dependent choices inside CV loop |
| Statistical validation | Pitfall 2: Window-level leakage | StratifiedGroupKFold + subject-level BA always |
| Statistical validation | Pitfall 7: Biased permutation null | Monitor effective_permutations, require >90% success |
| Interpretation | Pitfall 10: Medication confound | Report as limitation, check for medication metadata |

## Sources

- [Data leakage in deep learning studies of translational EEG](https://www.frontiersin.org/articles/10.3389/fnins.2024.1373515/full) — Brookshire et al., Frontiers in Neuroscience 2024. HIGH confidence.
- [Inflated prediction accuracy of neuropsychiatric biomarkers caused by data leakage in feature selection](https://www.nature.com/articles/s41598-021-87157-3) — Scientific Reports 2021. HIGH confidence.
- [Technical and clinical considerations for EEG-based biomarkers for MDD](https://www.nature.com/articles/s44184-023-00038-7) — Nature Mental Health 2023. HIGH confidence.
- [EEG is better left alone](https://www.nature.com/articles/s41598-023-27528-0) — Nature Scientific Reports 2023. MEDIUM confidence.
- [Autoreject: Automated artifact rejection for MEG and EEG data](https://www.researchgate.net/publication/311925766_Autoreject_Automated_artifact_rejection_for_MEG_and_EEG_data) — NeuroImage 2017. HIGH confidence.
- [A multi-modal open dataset for mental-disorder analysis (MODMA)](https://www.nature.com/articles/s41597-022-01211-x) — Scientific Data 2022. HIGH confidence.
- [Opportunities and Challenges for Clinical Practice in Detecting Depression Using EEG and ML](https://www.mdpi.com/1424-8220/25/2/409) — Sensors 2025. MEDIUM confidence.
