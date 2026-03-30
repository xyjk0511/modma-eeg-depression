# Delta Feature Ablation Summary

## 1. Experiment Setup

- **Pipeline:** StandardScaler + SVC(kernel=rbf, class_weight=balanced, random_state=42)
- **Subject filter:** exclude_high_fp (Phase 7)
- **CV:** 5-fold StratifiedGroupKFold
- **Statistical test:** scipy permutation_test, permutation_type=independent, n_resamples=9999

## 2. Feature Sets

| Strategy | Features | Dims |
|----------|----------|------|
| baseline | v3_subset only (676-dim) | 676 |
| delta_only | delta only (44-dim) | 44 |
| augmented | v3_subset + delta (720-dim) | 720 |

## 3. Results Comparison

| Strategy | BA | AUC | SEN | SPE | delta-BA | p-value |
|----------|-----|-----|-----|-----|----------|--------|
| baseline | 0.6913 | 0.7434 | 0.8033 | 0.5793 | - | - |
| delta_only | 0.5640 | 0.5892 | 0.4885 | 0.6394 | -0.1273 | 0.996032 |
| augmented | 0.6856 | 0.7455 | 0.7869 | 0.5843 | -0.0057 | 0.591270 |

## 4. Per-fold BA

| Fold | baseline | delta_only | augmented |
|------|----------|------------|----------|
| 1 | 0.7555 | 0.6412 | 0.7147 |
| 2 | 0.7218 | 0.5249 | 0.7260 |
| 3 | 0.6895 | 0.5792 | 0.6730 |
| 4 | 0.6516 | 0.5583 | 0.6557 |
| 5 | 0.6378 | 0.5163 | 0.6586 |

## 5. Decision

- **Recommendation:** reject
- **Rationale:** Delta features do not improve BA (0.6913 -> 0.6856)
- **Adopt delta features:** False

## 6. Notes

- Delta coherence computed via scipy.signal.coherence (no mne-connectivity dependency)
- NaN delta features (from failed coherence computation) filled with 0
- Phase 7 baseline (exclude_high_fp, v3_subset): BA=0.6913
