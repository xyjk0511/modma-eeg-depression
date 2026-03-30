# Error Analysis Summary

## 1. Experiment Setup

- **Pipeline:** StandardScaler + SVC(kernel=rbf, class_weight=balanced, random_state=42)
- **Features:** v3_subset (abs_bp + rel_bp + Hjorth, 676-dim, combined EO+EC)
- **CV:** 5-fold StratifiedGroupKFold
- **Subjects:** 1138 (MDD=305, nonMDD=833)
- **Threshold:** Youden's J optimal = 0.2469 (NOT default 0.5)

## 2. Confusion Matrix

| | Predicted nonMDD | Predicted MDD |
|---|---|---|
| **True nonMDD** | TN=440 (38.7%) | FP=393 (34.5%) |
| **True MDD** | FN=75 (6.6%) | TP=230 (20.2%) |

- **Sensitivity:** 0.7541
- **Specificity:** 0.5282
- **Balanced Accuracy:** 0.6412

## 3. Covariate Analysis

Bonferroni correction: 7 tests, threshold p < 0.007143

| Test | Comparison | Statistic | p-value | Significant? |
|------|-----------|-----------|---------|-------------|
| age | correct_vs_wrong | 94550.5 | 0.000000 | YES |
| age | FN_vs_TP | 8126.5 | 0.452731 | no |
| age | FP_vs_TN | 132710.5 | 0.000000 | YES |
| gender | correct_vs_wrong | 8.2 | 0.004233 | YES |
| gender | FN_vs_TP | 10.8 | 0.001012 | YES |
| gender | FP_vs_TN | 34.0 | 0.000000 | YES |
| education | correct_vs_wrong | 103841.5 | 0.000000 | YES |

## 4. FP Rate by nonMDD Indication (n >= 10)

| Indication | n | FP count | FP rate |
|-----------|---|----------|--------|
| INSOMNIA | 30 | 27 | 0.9000 |
| SMC | 86 | 57 | 0.6628 |
| TINNITUS | 29 | 19 | 0.6552 |
| OCD | 37 | 24 | 0.6486 |
| PARKINSON | 25 | 16 | 0.6400 |
| CHRONIC PAIN | 13 | 8 | 0.6154 |
| BURNOUT | 10 | 5 | 0.5000 |
| HEALTHY | 45 | 20 | 0.4444 |
| ADHD | 188 | 59 | 0.3138 |
| Dyslexia | 17 | 0 | 0.0000 |

Overall nonMDD FP rate: 0.4896

## 5. Key Findings

**Significant biases detected:**
- age_correct_vs_wrong
- age_FP_vs_TN
- gender_correct_vs_wrong
- gender_FN_vs_TP
- gender_FP_vs_TN
- education_correct_vs_wrong

**High FP-rate subgroups (above overall FP rate):**
- BURNOUT: FP rate=0.5, n=10
- CHRONIC PAIN: FP rate=0.6154, n=13
- INSOMNIA: FP rate=0.9, n=30
- OCD: FP rate=0.6486, n=37
- PARKINSON: FP rate=0.64, n=25
- SMC: FP rate=0.6628, n=86
- TINNITUS: FP rate=0.6552, n=29

**Feature-space distance analysis:**
- Misclassified subjects are farther from own-class centroid (median 363.98 vs 325.12, p=0.270495)

## 6. Recommendations for v3.0

1. Age is a significant confound. Consider age-stratified evaluation or age as a covariate in the model.
2. Gender bias detected. Investigate gender-specific EEG patterns.
3. High FP subgroups (BURNOUT, CHRONIC PAIN, INSOMNIA, OCD, PARKINSON, SMC, TINNITUS) may share EEG features with MDD. Consider multi-class or hierarchical classification.

## 7. Limitations

- SVM probability=True uses Platt scaling, which may be poorly calibrated on imbalanced data. Probabilities are used for ranking only.
- Optimal threshold selected on same OOF predictions used for analysis (acceptable for pattern analysis, not performance reporting).
- Dataset column too sparse (19.5% coverage) for site-level analysis.
