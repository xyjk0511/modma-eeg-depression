# Pre-Registration: Confirmatory Replication of MODMA MDD Classifier

**Date:** 2026-02-22
**Status:** LOCKED (git-tagged before any external data processing)

## Locked Pipeline

- **Features (5 dimensions):**
  1. parietal_theta_rel (relative theta power, parietal region, 4-8 Hz)
  2. parietal_alpha_rel (relative alpha power, parietal region, 8-13 Hz)
  3. alpha_asym_frontal (frontal alpha asymmetry, R-L / R+L)
  4. parietal_TBR (parietal theta/beta ratio, 4-8 Hz / 13-30 Hz)
  5. riem_central_temporal (Riemannian log-covariance, central-temporal 2x2)

- **Model:** LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
- **Scaler:** StandardScaler (fit on MODMA training data only)
- **Sample weighting:** Inverse subject-frequency balanced weights on training set

## Locked Preprocessing

- High-pass filter: 1.0 Hz
- Low-pass filter: 45.0 Hz
- Resample: 125 Hz
- Reference: average (after interpolation)
- QC amplitude threshold: 200 uV (no adjustment allowed)
- Window size: 10 seconds
- Window 0 skip: yes (filter transient)
- Crop duration: 60 seconds
- Bad channel interpolation: spherical spline if <= 25% channels are bad
- Per-window QC: discard window if > 12% of channels exceed 200 uV

## Locked Evaluation

- **Training set:** Full MODMA dataset (43 subjects: 18 MDD, 25 HC)
- **Test set:** External dataset (TDBRAIN, OpenNeuro ds003838)
- **Protocol:** Train on MODMA, test on external. No re-tuning.
- **Primary metric:** Subject-level balanced accuracy
- **Success criteria:**
  - Balanced accuracy > 0.60
  - Bootstrap 95% CI lower bound > 0.50
  - Permutation test p-value < 0.05
- **Permutations:** 1000 (subject-level label permutation on external test set)
- **Bootstrap CI:** 1000 resamples, percentile method

## Adaptation Whitelist

1. File format adapter (BDF via mne.io.read_raw_bdf)
2. Channel mapping (10-20 channels to 5 regions via STANDARD_1020_REGION_MAP)
3. Resampling from native rate to 125 Hz (with post-resample spectrum sanity check)

## Explicitly Forbidden

- Changing QC threshold (200 uV is locked)
- Changing feature set (5 dimensions are locked)
- Changing model or hyperparameters (LogisticRegression C=0.1 is locked)
- Changing evaluation criteria (BA>0.60, CI>0.50, p<0.05)
- Excluding subjects based on results
- Re-training or fine-tuning on external data

## Exploratory Phase 4 Results (reference)

- MODMA cross-validated BA: 0.613
- MODMA permutation p-value: 0.135 (not significant)
- Conclusion: exploratory/inconclusive

## Source Code Reference

- Training pipeline: modma_mdd_real_experiment.py
- Data adapter: external_data_adapter.py
- Feature extraction: extract_features() in modma_mdd_real_experiment.py
