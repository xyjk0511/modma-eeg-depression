# Feature Research

**Domain:** EEG-based MDD vs HC binary classification (resting-state, 128-channel, MODMA dataset)
**Researched:** 2026-02-22
**Confidence:** MEDIUM (training data + multiple web sources agree; no Context7 for neuroscience domain)

## Feature Landscape

### Table Stakes (Results Are Unreliable Without These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Relaxed QC thresholds (max_bad_channels ~15%)** | Current `max_bad_channels=3` for 128ch drops 80% of subjects. EGI-128 community standard is ~15% (~19 channels). With only 11/53 subjects, no classifier can learn. | LOW | Change from 3 to ~15-19. Single param. Most impactful fix. |
| **Bad channel interpolation (spherical spline)** | Instead of rejecting windows with bad channels, interpolate them. Preserves data. MNE `raw.interpolate_bads()`. Standard for high-density EEG. | MEDIUM | Detect bad channels via z-score on variance, interpolate, then proceed. |
| **Skip Window 0 (filter transient)** | First window after highpass has edge artifacts causing inflated bad-channel counts. Documented in PROJECT.md diagnostics. | LOW | Skip first `window_sec` after filtering. One-line fix. |
| **Highpass filter >= 0.5 Hz** | Removes DC offset and slow drifts. "EEG is better left alone" (Delorme 2023) confirms highpass + bad channel interpolation are the two steps that reliably help. | LOW | Already done at 0.5 Hz. Keep as-is. |
| **Subject-level evaluation** | Windows from same subject are correlated. Window-level eval inflates accuracy. Must use StratifiedGroupKFold + subject-level vote. | LOW | Already implemented correctly. Keep as-is. |
| **PSD band power (delta/theta/alpha/beta)** | Most replicated EEG biomarker for MDD. Every review includes these. Theta and alpha power differences are the most consistent findings. | LOW | Already implemented. 5 regions x 4 bands x 2 (abs+rel) = 40 dims. |
| **Frontal alpha asymmetry (FAA)** | Most studied single EEG biomarker for depression. Small effect sizes, debated reproducibility, but expected in any MDD pipeline. | LOW | Already implemented. 1 dim. |
| **Adaptive amplitude threshold** | Fixed 200uV is too strict for this dataset (EDF physical range has DC offsets). Need per-subject adaptive threshold (median + N*MAD). | MEDIUM | Replace fixed threshold with robust statistical threshold per subject. |

### Differentiators (Competitive Advantage in Accuracy)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Nonlinear entropy (SampEn, ApEn)** | MDD shows altered EEG complexity. Frontal alpha SampEn distinguishes mild depression from HC. HFD + SampEn achieved strong classification in multiple studies. | MEDIUM | ~20-40 new dims. Use `antropy` lib or ~20 lines numpy. |
| **Aperiodic (1/f) features via specparam** | MDD shows decreased aperiodic exponent (flatter 1/f slope), reflecting altered E/I balance. Separating periodic from aperiodic gives cleaner features than raw band power. | MEDIUM | `specparam` library. Exponent + offset per region. ~10 dims. |
| **Lempel-Ziv Complexity (LZC)** | Increased LZC in depression vs controls, strongest at frontal sites (p=0.0098 at F4). Captures complexity entropy may miss. | LOW | Binary median-split + LZ76 algorithm. ~5 dims. Pure numpy. |
| **Theta/beta ratio (TBR)** | Elevated TBR associated with attentional/emotional dysregulation in depression. Simple, clinically interpretable. | LOW | Ratio of existing band powers. ~5 dims. |
| **Gamma band power (30-45 Hz)** | Currently missing (bands stop at 30 Hz). Gamma abnormalities in MDD reported. Completes spectral picture. | LOW | Extend band dict. ~10 dims. Data already available (filter to 45 Hz). |
| **wPLI connectivity** | More robust than ImCoh against volume conduction. Standard in MNE-connectivity. | MEDIUM | Replace or supplement ImCoh. Same ~40 dims. Needs `mne-connectivity`. |
| **Phase-Amplitude Coupling (PAC)** | Theta-gamma PAC altered in depression, reflects disrupted cross-frequency communication. | HIGH | Modulation index per region. Computationally expensive. ~5-10 dims. |
| **Multiscale nonlinear fusion** | Cross-subject MDD detection achieved 72-85% accuracy. High-frequency scale LZC particularly effective. | HIGH | Features at multiple coarse-grained scales. Multiplies feature count. |

### Anti-Features (Things to Deliberately NOT Do)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Deep learning (CNN/Transformer)** | Papers report 95-99% accuracy | With ~40 subjects, massive overfit. Those accuracies use within-subject splits. LOSO drops to ~37%. | SVM/LR/LightGBM with regularization. |
| **ICA artifact removal** | Standard preprocessing | Delorme 2023 showed ICA rejection didn't reliably improve classification. Interpolation before ICA creates ghost ICs from rank deficiency. | Bad channel interpolation + amplitude rejection. |
| **Aggressive bandpass (1-40 Hz)** | Remove noise | Loses gamma (30-45 Hz) and sub-delta. Current 0.5-45 Hz is appropriate. | Keep 0.5-45 Hz. |
| **Very short windows (< 5s)** | More training samples | Poor spectral resolution for delta. More windows don't help -- subject-level eval negates the benefit. | Keep 10-sec windows. |
| **Source localization (eLORETA)** | Better spatial specificity | Requires head model + digitized positions. MODMA may lack these. Massive complexity, uncertain gain with small N. | Sensor-space features with region averaging. |
| **Complex feature selection (mRMR, GA)** | Optimal subset | With N~40, complex selection overfits. | PCA(0.95) or SelectKBest(k=20-30). |
| **Window-level accuracy reporting** | Higher numbers | Inflated by within-subject correlation. Not valid generalization estimate. | Subject-level balanced accuracy + permutation test. |

## Feature Dependencies

```
[Bad Channel Interpolation]
    └──requires──> [Relaxed QC Thresholds]
                       └──enables──> [More Subjects (35-40)]
                                         └──enables──> [Meaningful Classification]

[Skip Window 0]
    └──enhances──> [QC Pass Rate]

[Aperiodic Features (specparam)]
    └──requires──> [PSD Computation] (already exists)
    └──conflicts──> [Raw Band Power] (aperiodic-corrected peaks replace raw)

[Nonlinear Features (SampEn, LZC)]
    └──independent of──> [PSD Features] (complementary representations)

[wPLI Connectivity]
    └──enhances──> [ImCoh Connectivity] (can replace or supplement)

[Gamma Band]
    └──requires──> [Low-pass >= 45 Hz] (already satisfied)
```

### Dependency Notes

- **Bad Channel Interpolation requires Relaxed QC:** Interpolation is the mechanism that makes relaxed thresholds safe -- you fix bad channels instead of rejecting windows.
- **Aperiodic features conflict with raw band power:** Once you decompose PSD into periodic + aperiodic, the corrected periodic peaks are more meaningful. Can keep both but partially redundant.
- **Nonlinear features are independent:** SampEn and LZC operate on time-domain signal, complementary to frequency-domain PSD.
- **More subjects enables meaningful CV:** With 11 subjects, 5-fold CV has ~2 per fold. With 40, you get ~8 per fold -- much more stable.

## MVP Definition

### Launch With (v1) -- Fix QC to Get Enough Subjects

- [x] Relax `max_bad_channels` from 3 to ~15 (12% of 128) -- **most critical fix**
- [x] Skip Window 0 (filter transient) -- **easy win**
- [ ] Add bad channel interpolation (spherical spline via MNE) -- **preserves data quality**
- [ ] Adaptive amplitude threshold (median + 5*MAD per subject) -- **replaces brittle 200uV**
- [ ] Target: 35-40 subjects retained, BA > 0.60

### Add After Validation (v1.x) -- Feature Engineering

- [ ] Nonlinear features: SampEn per region per band (~20 dims)
- [ ] Aperiodic features via specparam: exponent + offset per region (~10 dims)
- [ ] LZC per region (~5 dims)
- [ ] Gamma band power (~10 dims)
- [ ] Theta/beta ratio (~5 dims)

### Future Consideration (v2+)

- [ ] wPLI connectivity (replace ImCoh) -- defer until v1.x features evaluated
- [ ] Phase-amplitude coupling -- high complexity, defer
- [ ] Multiscale entropy -- defer until single-scale entropy proven
- [ ] Microstate features -- requires separate analysis pipeline

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Relax max_bad_channels to ~15 | HIGH | LOW | **P1** |
| Skip Window 0 | HIGH | LOW | **P1** |
| Bad channel interpolation | HIGH | MEDIUM | **P1** |
| Adaptive amplitude threshold | HIGH | MEDIUM | **P1** |
| Sample Entropy (per region) | MEDIUM | MEDIUM | **P2** |
| Aperiodic features (specparam) | MEDIUM | MEDIUM | **P2** |
| Lempel-Ziv Complexity | MEDIUM | LOW | **P2** |
| Gamma band power | LOW | LOW | **P2** |
| Theta/beta ratio | LOW | LOW | **P2** |
| wPLI connectivity | MEDIUM | MEDIUM | **P3** |
| Phase-amplitude coupling | LOW | HIGH | **P3** |
| Multiscale entropy | LOW | HIGH | **P3** |

**Priority key:**
- P1: Must have -- fixes the core QC bottleneck (11 -> 40 subjects)
- P2: Should have -- adds discriminative features once sample size is fixed
- P3: Nice to have -- advanced features for marginal gains

## Competitor Feature Analysis (Literature Benchmarks)

| Approach | Typical BA (cross-subject) | Features Used | Our Approach |
|----------|---------------------------|---------------|--------------|
| PSD + SVM (traditional) | 65-80% | Band power, asymmetry | Already have; need more subjects |
| Nonlinear + ML | 70-85% | SampEn, HFD, LZC + SVM/RF | Add SampEn + LZC (P2) |
| Aperiodic + periodic | ~75% | FOOOF exponent + corrected peaks | Add specparam (P2) |
| Entropy + microstate + DL | 96-99% (inflated) | Shannon/SampEn + CNN | NOT comparable -- within-subject eval |
| Multiscale nonlinear | 72-85% | Multi-scale LZC + fusion | Consider for v2 |
| Cross-subject realistic | 60-75% | Various | Our realistic target range |

**Note:** Many papers report 90%+ but use within-subject splits or CV without subject grouping. Realistic cross-subject BA with proper evaluation is 60-80%.

## Sources

- [Depression Detection and Diagnosis Based on EEG Analysis: A Comprehensive Review (2025)](https://www.mdpi.com/2075-4418/15/2/210/html) -- MEDIUM confidence
- [Technical and clinical considerations for EEG-based biomarkers for MDD (Nature, 2023)](https://www.nature.com/articles/s44184-023-00038-7) -- HIGH confidence
- [EEG is better left alone (Delorme, Nature Sci Rep 2023)](https://www.nature.com/articles/s41598-023-27528-0) -- HIGH confidence
- [Aperiodic and Periodic EEG Components in MDD Classification (2024)](https://www.mdpi.com/1424-8220/24/18/6103) -- MEDIUM confidence
- [Frontal Alpha Complexity of Depression Patients (2020)](https://www.hindawi.com/journals/jhe/2020/8854725/) -- MEDIUM confidence
- [Lempel Ziv Complexity of EEG in Depression](https://www.researchgate.net/publication/278659644_Lempel_Ziv_Complexity_of_EEG_in_Depression) -- MEDIUM confidence
- [EGI-128 bad channel threshold ~15% (ResearchGate)](https://www.researchgate.net/post/How_many_electrodes_can_be_interpolated_in_an_EEG_recording_while_maintaining_integrity_of_the_data) -- MEDIUM confidence
- [Cross-subject MDD detection via multiscale nonlinear analysis (2025)](https://magazine.ingentium.com/2025/07/15/a-cross-subject-mdd-detection-approach-based-on-multiscale-nonlinear-analysis-in-resting-state-eeg/) -- MEDIUM confidence
- [Aperiodic activity in OCD and MDD (2025)](https://magazine.ingentium.com/2025/11/19/analysis-of-aperiodic-activity-in-obsessive-compulsive-disorder-and-major-depression/) -- LOW confidence
- [Opportunities and Challenges for Depression Detection Using EEG and ML (2025)](https://www.mdpi.com/1424-8220/25/2/409) -- MEDIUM confidence
- [EEG-based MDD recognition by neural oscillation and asymmetry (2024)](https://www.frontiersin.org/articles/10.3389/fnins.2024.1362111/full) -- MEDIUM confidence
- [EEG machine learning with HFD and SampEn for depression detection](https://www.researchgate.net/publication/323846338_EEG_machine_learning_with_Higuchi_fractal_dimension_and_Sample_Entropy_as_features_for_successful_detection_of_depression) -- MEDIUM confidence

---
*Feature research for: EEG MDD vs HC classification (MODMA 128-channel resting-state)*
*Researched: 2026-02-22*
