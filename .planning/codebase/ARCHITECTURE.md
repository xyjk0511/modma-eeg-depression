# Architecture

**Analysis Date:** 2026-02-22

## Pattern Overview

**Overall:** Layered pipeline architecture for EEG signal processing and MDD (Major Depressive Disorder) classification.

**Key Characteristics:**
- Data loading → preprocessing → feature extraction → classification
- Modular feature computation (PSD, connectivity, Riemannian, asymmetry)
- Scikit-learn pipeline for standardized model training
- Support for both simulated and real BIDS-format EEG data
- Parallelized permutation testing with joblib

## Layers

**Data Loading Layer:**
- Purpose: Load EEG data from BIDS-formatted directories or generate synthetic data
- Location: `load_windows()`, `load_real_modma_data()` in `/d/eeg/modma_mdd_real_experiment.py`; `load_modma_data()` in `/d/eeg/modma_mdd_classification.py`
- Contains: EDF file reading, BIDS participant parsing, windowing logic
- Depends on: MNE-Python, pandas, glob
- Used by: Feature extraction layer

**Preprocessing Layer:**
- Purpose: Filter, resample, reference, and quality-check raw EEG signals
- Location: Within `load_windows()` and `_load_subject()` in `/d/eeg/modma_mdd_real_experiment.py`
- Contains: High-pass filtering (0.5 Hz), low-pass filtering (45 Hz), resampling (125 Hz), average referencing, amplitude-based QC
- Depends on: MNE-Python signal processing
- Used by: Feature extraction layer

**Feature Extraction Layer:**
- Purpose: Compute 144-dimensional feature vectors from preprocessed EEG windows
- Location: `extract_features()` in `/d/eeg/modma_mdd_real_experiment.py` (lines 215-284)
- Contains:
  - Per-channel PSD (power spectral density) in delta/theta/alpha/beta bands
  - Region-aggregated features (std, absolute/relative band power)
  - Differential entropy (DE) features
  - Hjorth features (activity, mobility, complexity)
  - Lateral asymmetry (frontal/temporal L-R)
  - Alpha asymmetry (frontal)
  - Imaginary coherence (connectivity, 40 dims)
  - Riemannian log-covariance features (15 dims)
- Depends on: scipy.signal (welch, csd), numpy
- Used by: Classification layer

**Classification Layer:**
- Purpose: Train and evaluate binary classifiers (MDD vs HC)
- Location: `build_feature_model_pipeline()`, cross-validation logic in `/d/eeg/modma_mdd_real_experiment.py`
- Contains: SVM, Logistic Regression, LightGBM with optional PCA/SelectKBest
- Depends on: scikit-learn, lightgbm (optional)
- Used by: Evaluation layer

**Evaluation Layer:**
- Purpose: Perform cross-validation, permutation testing, and generate reports
- Location: `run_evaluation()` in `/d/eeg/run_modma_real.py`; permutation loop in `/d/eeg/modma_mdd_real_experiment.py`
- Contains: StratifiedGroupKFold, GridSearchCV, permutation testing with joblib parallelization
- Depends on: scikit-learn, joblib
- Used by: Main entry points

## Data Flow

**Real Data Pipeline:**

1. Load BIDS dataset → parse participants.tsv
2. For each subject: locate EDF file, read with MNE
3. Crop to 60s, filter (0.5-45 Hz), resample to 125 Hz, average reference
4. Segment into 10-second windows
5. Quality check: reject windows with >3 channels exceeding 200 µV
6. Extract 144-dim features per window
7. Train/test split at subject level (StratifiedGroupKFold)
8. Fit pipeline (scaler → optional reducer → classifier)
9. Cross-validate and report metrics

**State Management:**
- Window-level labels: 0 (HC) or 1 (MDD)
- Subject-level grouping: preserved via `groups` array to prevent data leakage
- Feature matrix: (n_windows, 144) after extraction
- Immutable: all arrays created fresh, no in-place mutations

## Key Abstractions

**Region Mapping:**
- Purpose: Map EEG channels to brain regions (frontal, central, temporal, parietal, occipital)
- Examples: `build_region_indices()`, `EGI128_REGION_MAP`, `STANDARD_1020_REGION_MAP` in `/d/eeg/modma_mdd_real_experiment.py`
- Pattern: Auto-detect EGI-128 vs 10-20 system; return dict of region → channel indices

**Feature Computation Functions:**
- Purpose: Encapsulate individual feature families
- Examples: `compute_de_features()`, `compute_hjorth_features()`, `compute_connectivity_features()`, `compute_riemannian_features()` in `/d/eeg/modma_mdd_real_experiment.py`
- Pattern: Each returns (feature_array, feature_names) tuple for composability

**Pipeline Builder:**
- Purpose: Construct scikit-learn Pipeline with configurable reducer and classifier
- Examples: `build_feature_model_pipeline()` in `/d/eeg/modma_mdd_real_experiment.py`
- Pattern: Steps list with optional PCA/SelectKBest; supports SVM, LogisticRegression, LGBMClassifier

## Entry Points

**`run_modma_real.py`:**
- Location: `/d/eeg/run_modma_real.py`
- Triggers: `python run_modma_real.py`
- Responsibilities: Load real BIDS data, extract features, run StratifiedGroupKFold CV, report accuracy

**`modma_mdd_real_experiment.py` (main):**
- Location: `/d/eeg/modma_mdd_real_experiment.py` (lines 900+)
- Triggers: `python modma_mdd_real_experiment.py --bids-root <path> [options]`
- Responsibilities: Full pipeline with permutation testing, grid search, detailed reporting

**`modma_mdd_classification.py` (demo):**
- Location: `/d/eeg/modma_mdd_classification.py`
- Triggers: `python modma_mdd_classification.py`
- Responsibilities: Synthetic data demo with Alpha Asymmetry + SVM

## Error Handling

**Strategy:** Explicit validation at data boundaries; raise ValueError/FileNotFoundError with descriptive messages.

**Patterns:**
- File existence checks: `if not os.path.exists(participants_file)` → FileNotFoundError
- Data sufficiency checks: `if not all_epochs or ch_names_ref is None` → ValueError
- Subject balance validation: `validate_subject_class_counts()` ensures ≥2 subjects per class
- Window availability: `validate_post_qc_availability()` ensures ≥3 windows per subject post-QC
- Type validation: `if not isinstance(max_subjects, int)` → TypeError

## Cross-Cutting Concerns

**Logging:**
- Framework: Python `logging` module
- Pattern: `logger.info()` for progress, `logger.warning()` for skipped subjects, `logger.error()` for exceptions

**Validation:**
- Input: BIDS path, max_subjects (even, ≥2), window size, resample frequency
- Output: Feature matrix shape, label distribution, group balance

**Authentication:**
- Not applicable (file-based data)

---

*Architecture analysis: 2026-02-22*
