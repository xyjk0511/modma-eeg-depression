# Ensemble Modeling Summary

## 1. Experiment Setup

- **SVM:** StandardScaler + SVC(kernel=rbf, class_weight=balanced, random_state=42)
- **XGB:** StandardScaler + XGBClassifier(scale_pos_weight, nested CV tuning)
- **Ensemble:** Weighted soft voting (w_svm * P_svm + w_xgb * P_xgb)
- **Subject filter:** exclude_high_fp (Phase 7)
- **Features:** v3_subset (abs_bp + rel_bp + Hjorth, 676d)
- **Outer CV:** 5-fold StratifiedGroupKFold
- **Inner CV:** 3-fold (XGB tuning + weight search)
- **XGB search:** RandomizedSearchCV, n_iter=50
- **Statistical test:** permutation_test, n_resamples=9999, alternative=greater

## 2. Results Comparison

| Strategy | BA | AUC | SEN | SPE | delta-BA | p-value |
|----------|-----|-----|-----|-----|----------|--------|
| svm_only | 0.6913 | 0.7434 | 0.8033 | 0.5793 | - | - |
| xgb_only | 0.5950 | 0.7285 | 0.3803 | 0.8097 | -0.0963 | 0.992063 |
| ensemble | 0.5983 | 0.7314 | 0.3836 | 0.8130 | -0.0930 | 0.992063 |

## 3. Per-fold BA

| Fold | svm_only | xgb_only | ensemble |
|------|----------|----------|----------|
| 1 | 0.7555 | 0.5643 | 0.5725 |
| 2 | 0.7218 | 0.6666 | 0.6708 |
| 3 | 0.6895 | 0.6212 | 0.6130 |
| 4 | 0.6516 | 0.5641 | 0.5805 |
| 5 | 0.6378 | 0.5587 | 0.5547 |

## 4. Ensemble Weights per Fold

| Fold | w_svm | w_xgb |
|------|-------|-------|
| 1 | 0.3 | 0.7 |
| 2 | 0.7 | 0.3 |
| 3 | 0.4 | 0.6 |
| 4 | 0.5 | 0.5 |
| 5 | 0.4 | 0.6 |

## 5. Decision

- **Adopt ensemble:** False
- **Rationale:** Ensemble does not improve BA (0.6913 -> 0.5983)
- **Final model:** svm_only
- **Final BA:** 0.6913
