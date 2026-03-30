# Permutation Test Summary

## 1. Experiment Setup

- **Model:** SVM (kernel=rbf, class_weight=balanced, random_state=42)
- **Pipeline:** StandardScaler + SVC (no SMOTE, Phase 2 locked decision)
- **Feature sets:** v3_subset (abs_bp + rel_bp + Hjorth, 676-dim), full (all features, 992-dim)
- **Condition:** combined (EO+EC)
- **Subjects:** 1138
- **Permutations:** 1000
- **CV:** 5-fold StratifiedGroupKFold (group-level, no subject leakage)
- **Permutation level:** Group-level label shuffling
- **p-value correction:** Phipson & Smyth (2010) +1 correction
- **Multiple comparisons:** Bonferroni correction (2 comparisons, threshold p < 0.025)

## 2. Results

| Feature Set | Observed BA | Mean Null BA | Std Null | p-value | Significant? | Cohen's d |
|-------------|-------------|-------------|----------|---------|-------------|----------|
| v3_subset | 0.6318 | 0.5007 | 0.0209 | 0.0010 | YES | 6.2798 |
| full | 0.6147 | 0.5012 | 0.0206 | 0.0010 | YES | 5.4952 |

## 3. Significance Determination

Bonferroni correction applied for 2 comparisons: alpha_corrected = 0.05 / 2 = 0.025

- **v3_subset:** p = 0.0010 < 0.025 -- **statistically significant**. The observed BA (0.6318) is unlikely under the null hypothesis.
- **full:** p = 0.0010 < 0.025 -- **statistically significant**. The observed BA (0.6147) is unlikely under the null hypothesis.

## 4. Null Distribution 95% CI

- **v3_subset:** [0.4573, 0.5408]
- **full:** [0.4607, 0.5422]

