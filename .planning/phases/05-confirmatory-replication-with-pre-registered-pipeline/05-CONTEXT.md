# Phase 5: Confirmatory Replication — Context Decisions

**Date:** 2026-02-22
**Areas discussed:** 4/4

## 1. Data Source

- **Primary:** External public EEG dataset (needs research to identify best candidate)
- **Montage:** 10-20 system acceptable (19-64 channels)
- **Size:** >= 50 subjects, >= 25 per group (MDD vs HC)
- **Paradigm:** Eyes-closed resting state only
- **Fallback:** MODMA leave-N-out allowed but labeled as exploratory/internal validation only
  - External data validation is the primary goal
  - Fallback results must NOT be used to modify features/hyperparameters

## 2. Pre-registration Scope

- **Scope:** End-to-end lockdown (pipeline + criteria + preprocessing params)
- **Format:** Repo internal document (markdown), git tag locked
- **Adaptation whitelist:** File format adapter, channel mapping, resampling only
- **Explicitly forbidden:** Changing QC thresholds, features, model/hyperparams, evaluation protocol
- **Timing:** Tag BEFORE first run; data adapter scripts must be included in tag

## 3. Evaluation Protocol

- **Criteria:** Same as Phase 4 (BA > 0.60, CI lower > 0.50, p < 0.05)
- **Primary mode:** Train-MODMA / Test-external
  - Rule 1: Train on full MODMA (43 subjects), test on external
  - Rule 2: No re-tuning on external data
  - Rule 3: Subject-level metrics on external test set
- **Permutation:** 1000 iterations on external test set
- **Reporting:** Complete report regardless of outcome (positive or negative)

## 4. Data Compatibility

- **Channel mapping:** Region-level mapping (frontal/central/temporal/parietal/occipital)
  - Each region uses available channels from the target dataset
  - No requirement for exact channel name match
- **Sampling rate:** Resample to 125Hz allowed (whitelist) but requires spectrum sanity check
- **Reference:** Unified re-reference to average (same as MODMA pipeline)
- **QC threshold:** Strictly locked at 200uV — no adjustment allowed; report as-is
