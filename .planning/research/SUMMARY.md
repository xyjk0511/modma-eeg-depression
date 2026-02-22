# Project Research Summary

**Project:** MODMA EEG MDD vs HC Classification Pipeline
**Domain:** EEG-based depression biomarker extraction and binary classification
**Researched:** 2026-02-22
**Confidence:** MEDIUM

## Executive Summary

This is a research-grade EEG classification pipeline targeting MDD vs HC binary classification on the MODMA dataset (53 subjects, 128-channel EGI HydroCel, resting-state). The core problem is not a modeling problem -- it is a QC problem. The current pipeline retains only 11/53 subjects due to a hardcoded max_bad_channels=3 threshold that is 5x stricter than the EGI-128 community standard (~15%). With 11 subjects, no classifier can learn generalizable patterns. The single highest-leverage fix is relaxing QC thresholds and adding bad channel interpolation, which should bring retained subjects to 35-40.

The recommended approach is a two-stage upgrade: first fix QC to get adequate sample size (Phase 1), then add discriminative features once the pipeline is validated on sufficient subjects (Phase 2). The architecture should be refactored from a monolithic script into 7 focused modules (config, regions, io, qc, features, classify, evaluate) with a frozen dataclass config and .npz caching. New libraries -- autoreject (data-driven epoch repair), pyprep (PREP bad channel detection), and mne-icalabel (automated ICA artifact removal) -- address the QC bottleneck systematically.

The key risks are: (1) differential QC dropout creating artificial class imbalance between MDD and HC groups, (2) data leakage via window-level CV splitting (already mitigated in current code -- must not regress), and (3) overfitting with 144 features on ~40 effective subjects. Realistic cross-subject balanced accuracy with proper evaluation is 60-75%; papers reporting 90%+ use within-subject splits and are not comparable.

## Key Findings

### Recommended Stack

The existing stack (MNE 1.11, scikit-learn 1.8, pyriemann 0.10, joblib) is solid and should not be replaced. Three new libraries directly address the 80% subject dropout: autoreject 0.4.3 for data-driven per-epoch bad channel repair (replaces hardcoded 200uV threshold), pyprep 0.5.0 for PREP-standard bad channel detection on continuous data, and mne-icalabel 0.8.1 for automated ICA artifact classification (removes ocular/muscular artifacts that corrupt FAA). A fourth library, mne-connectivity 0.7.0, replaces the hand-rolled ImCoh implementation with validated wPLI support.

Recommended preprocessing order: highpass 1 Hz -> pyprep bad channel detection -> average re-reference -> ICA (extended infomax) -> mne-icalabel artifact removal -> bandpass [1,45] Hz -> epoch -> skip Window 0 -> autoreject local repair -> feature extraction.

**Core technologies:**
- autoreject 0.4.3: per-epoch bad channel interpolation -- replaces brittle 200uV threshold, maximizes data retention
- pyprep 0.5.0: PREP-standard bad channel detection -- multi-criteria (correlation, RANSAC, deviation) vs amplitude-only
- mne-icalabel 0.8.1: automated ICA component classification -- removes blink/muscle artifacts that corrupt FAA
- mne-connectivity 0.7.0: validated spectral connectivity -- adds wPLI, fixes ImCoh formula risk

### Expected Features

The P1 features are all QC fixes, not new signal features. Relaxing max_bad_channels from 3 to ~15 (12% of 128 channels) is the single most impactful change. Bad channel interpolation via spherical spline makes relaxed thresholds safe. Skipping Window 0 removes filter transient artifacts. An adaptive amplitude threshold (median + 5*MAD per subject) replaces the dataset-inappropriate fixed 200uV.

Once 35-40 subjects are retained, P2 features add discriminative signal: nonlinear entropy (SampEn per region, ~20 dims), aperiodic features via specparam (~10 dims), Lempel-Ziv Complexity (~5 dims), gamma band power (~10 dims), and theta/beta ratio (~5 dims).

**Must have (P1 -- fixes QC bottleneck):**
- Relax max_bad_channels to ~15 -- 11 subjects is not enough to train any classifier
- Bad channel interpolation (spherical spline) -- preserves data instead of rejecting windows
- Skip Window 0 -- removes filter transient artifacts confirmed in PROJECT.md diagnostics
- Adaptive amplitude threshold -- fixed 200uV is miscalibrated for MODMA EDF scaling anomalies

**Should have (P2 -- adds discriminative features):**
- Sample Entropy per region -- MDD shows altered EEG complexity, frontal alpha SampEn discriminates
- Aperiodic features (specparam) -- MDD shows decreased 1/f exponent, cleaner than raw band power
- Lempel-Ziv Complexity -- frontal LZC elevated in depression (p=0.0098 at F4)
- Gamma band power -- currently missing, completes spectral picture
- Theta/beta ratio -- clinically interpretable, low implementation cost

**Defer (v2+):**
- wPLI connectivity -- defer until P2 features evaluated
- Phase-amplitude coupling -- high complexity, uncertain gain at N~40
- Multiscale entropy -- defer until single-scale entropy proven

### Architecture Approach

The pipeline should be refactored from a monolithic script into 7 modules with clear input/output contracts. Each module has a distinct iteration cadence: qc.py is tuned frequently, features.py is extended incrementally, classify.py is rarely changed. A frozen dataclass config eliminates hidden global state. .npz caching after EDF loading reduces iteration time from ~2 min to <1 sec.

**Major components:**
1. config.py -- frozen dataclass, single source of truth for all parameters
2. io.py + qc.py -- EDF loading with .npz cache; QC masking with bad channel interpolation
3. features.py -- feature registry pattern: (X, sfreq, region_idx) -> (features, names)
4. classify.py -- nested CV with StratifiedGroupKFold, subject-level aggregation
5. evaluate.py -- permutation test on saved CV results (features extracted once, not 1000x)

### Critical Pitfalls

1. **Differential QC dropout creating artificial class imbalance** -- after relaxing thresholds, compare MDD vs HC retention rates; if one group loses >20% more subjects, the classifier may learn noise quality rather than neural signal.

2. **Data leakage via window-level CV splitting** -- already mitigated with StratifiedGroupKFold + subject-level BA aggregation. Must not regress during refactoring.

3. **Feature selection / hyperparameter tuning outside CV** -- all data-dependent decisions must stay inside the CV loop. Refactoring classify.py is the highest-risk step.

4. **Average reference amplifying bad channels** -- apply pyprep bad channel detection and interpolation BEFORE average re-referencing, not after.

5. **Overfitting with 144 features on ~40 subjects** -- apply PCA(0.95) or SelectKBest(k=20-30) inside CV. Target <10 effective features for 40 subjects.

## Implications for Roadmap

### Phase 1: QC Repair and Pipeline Refactor
**Rationale:** Without 35+ subjects, no downstream work is meaningful. QC fixes are the critical path.
**Delivers:** 35-40 retained subjects, modular codebase, .npz caching, qc_report.csv with group retention analysis
**Addresses:** Relax max_bad_channels, skip Window 0, bad channel interpolation, adaptive amplitude threshold, average reference ordering
**Avoids:** Pitfall 1 (differential dropout), Pitfall 4 (filter transient), Pitfall 6 (amplitude miscalibration), Pitfall 8 (average reference noise spread)
**Research flag:** Standard patterns for autoreject + pyprep; ICLabel on EGI-128 montage needs empirical validation on 2-3 subjects first

### Phase 2: Feature Engineering
**Rationale:** Only after Phase 1 validates 35+ subjects retained and baseline BA > 0.60 does feature engineering make sense.
**Delivers:** Expanded feature set (SampEn, aperiodic, LZC, gamma, TBR), ablation study showing which features contribute
**Uses:** antropy or numpy for SampEn/LZC; specparam for aperiodic; feature registry pattern
**Avoids:** Pitfall 5 (overfitting -- PCA inside CV), Pitfall 9 (volume conduction -- check ImCoh plausibility)
**Research flag:** specparam integration may need phase research (conflicts with raw band power features)

### Phase 3: Statistical Validation and Reporting
**Rationale:** Permutation test and bootstrap CI are the scientific validity gate. Must run after feature set is finalized.
**Delivers:** p-value, bootstrap 95% CI on BA, stopping criteria assessment, final metrics.json and plots
**Implements:** evaluate.py with features-extracted-once optimization (not 1000x re-extraction)
**Avoids:** Pitfall 2 (window-level leakage), Pitfall 7 (biased permutation null -- monitor effective_permutations > 900/1000)
**Research flag:** Standard patterns -- permutation test methodology is well-established

### Phase Ordering Rationale

- Phase 1 before Phase 2: sample size is the binding constraint; feature engineering on 11 subjects is invalid
- Architecture refactor in Phase 1: modular structure enables safe QC iteration and independent component testing
- ICA artifact removal in Phase 1: eye blink artifacts in frontal channels directly corrupt FAA -- the primary MDD biomarker
- Evaluation in Phase 3: permutation test must run on the final feature set to avoid inflating p-values

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (ICLabel on EGI-128):** ICLabel trained on standard 10-20 montage; EGI HydroCel uses non-standard channel names (E1, E2...). Verify on 2-3 subjects before full run.
- **Phase 1 (pyprep RANSAC on MODMA EDF):** MODMA EDF files have documented DC offset anomalies. Must apply highpass BEFORE pyprep.
- **Phase 2 (specparam integration):** aperiodic-corrected periodic peaks partially conflict with raw band power features. Decide whether to replace or supplement.

Phases with standard patterns (skip research-phase):
- **Phase 1 (autoreject local):** well-documented, API is straightforward, works on arbitrary epoch content
- **Phase 2 (SampEn/LZC):** pure numpy implementations, well-documented algorithms
- **Phase 3 (permutation test):** already implemented correctly, only needs optimization

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Versions verified via PyPI; ICLabel on EGI-128 montage is LOW confidence -- untested combination |
| Features | MEDIUM | Multiple sources agree on PSD/entropy/aperiodic; realistic BA targets (60-75%) well-supported |
| Architecture | HIGH | MNE design philosophy + reproducible EEG group study literature; patterns are established |
| Pitfalls | HIGH | Data leakage pitfalls backed by high-confidence peer-reviewed sources (Brookshire 2024, Saarschmidt 2021) |

**Overall confidence:** MEDIUM

### Gaps to Address

- **ICLabel on EGI-128 montage:** needs empirical validation on first 2-3 subjects. If predictions are implausible, fall back to amplitude-only artifact rejection.
- **autoreject on resting-state windows:** consensus parameter may need tuning for 10-second windows. Test with defaults first.
- **MODMA medication metadata:** check participants.tsv for medication status before Phase 3 interpretation. Report as limitation if unavailable.
- **Realistic BA target:** if Phase 1 yields BA < 0.55 with 35+ subjects, investigate MODMA label quality or medication confound before adding more features.

## Sources

### Primary (HIGH confidence)
- https://autoreject.github.io/stable/index.html -- local/global mode, resting-state use
- https://pyprep.readthedocs.io/en/stable/ -- NoisyChannels API, RANSAC
- https://mne.tools/mne-icalabel/stable/ -- ICLabel requirements
- Brookshire et al., Frontiers in Neuroscience 2024 -- data leakage in EEG DL studies
- Saarschmidt et al., Scientific Reports 2021 -- feature selection leakage inflates BA 10-20%
- Technical considerations for EEG biomarkers in MDD, Nature Mental Health 2023

### Secondary (MEDIUM confidence)
- EEG is better left alone, Delorme Nature Sci Rep 2023 -- highpass + interpolation are the two reliable preprocessing steps
- Depression Detection Review, MDPI Diagnostics 2025 -- feature landscape
- Aperiodic and Periodic EEG in MDD, MDPI Sensors 2024 -- specparam for depression
- EGI-128 interpolation threshold ~15%, ResearchGate

### Tertiary (LOW confidence)
- Aperiodic activity in OCD and MDD, 2025 -- aperiodic exponent in MDD, needs validation

---
*Research completed: 2026-02-22*
*Ready for roadmap: yes*
