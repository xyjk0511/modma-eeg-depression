# Class Imbalance Strategy Sweep — Summary

## 1. Experiment Setup

- **Strategies:** none, class_weight, smote, smote_cw
- **Feature sets:** v3_subset (abs_bp + rel_bp + Hjorth), full (all 992 features)
- **Seeds:** [42, 0, 123]
- **CV:** 5-fold StratifiedGroupKFold (no subject leakage)
- **Model:** SVM (kernel=rbf, fixed params, no GridSearch)
- **Condition:** combined (EO+EC concatenated, 992-dim)
- **Subjects:** 1138
- **Label counts:** {'MDD': 305, 'nonMDD': 833}
- **SMOTE:** k=5 with fallback to k=3, then no SMOTE (applied inside CV fold only)

## 2. Results

### Feature set: v3_subset

| Strategy | BA mean±std | AUC mean | SEN mean | SPE mean |
|----------|-------------|----------|----------|----------|
| No resampling | 0.5024±0.0000 | 0.6510 | 0.0131 | 0.9916 |
| class_weight=balanced | 0.6318±0.0000 | 0.6887 | 0.6885 | 0.5750 |
| SMOTE | 0.6155±0.0019 | 0.6744 | 0.5913 | 0.6399 |
| SMOTE + class_weight | 0.6155±0.0019 | 0.6744 | 0.5913 | 0.6399 |

### Feature set: full

| Strategy | BA mean±std | AUC mean | SEN mean | SPE mean |
|----------|-------------|----------|----------|----------|
| No resampling | 0.4958±0.0000 | 0.6584 | 0.0000 | 0.9916 |
| class_weight=balanced | 0.6147±0.0000 | 0.6746 | 0.6459 | 0.5834 |
| SMOTE | 0.5984±0.0026 | 0.6661 | 0.5170 | 0.6799 |
| SMOTE + class_weight | 0.5984±0.0026 | 0.6661 | 0.5170 | 0.6799 |

## 3. Constraint Check (SEN>=0.40 AND SPE>=0.40)

| Strategy | Feature set | SEN mean | SPE mean | Passes? | Any seed fails? |
|----------|-------------|----------|----------|---------|----------------|
| No resampling | v3_subset | 0.0131 | 0.9916 | NO | YES |
| class_weight=balanced | v3_subset | 0.6885 | 0.5750 | YES | no |
| SMOTE | v3_subset | 0.5913 | 0.6399 | YES | no |
| SMOTE + class_weight | v3_subset | 0.5913 | 0.6399 | YES | no |
| No resampling | full | 0.0000 | 0.9916 | NO | YES |
| class_weight=balanced | full | 0.6459 | 0.5834 | YES | no |
| SMOTE | full | 0.5170 | 0.6799 | YES | no |
| SMOTE + class_weight | full | 0.5170 | 0.6799 | YES | no |

## 4. Recommended Strategy

**Recommended strategy:** `class_weight`

**Feature set:** `v3_subset`

**Rationale:** strategy=class_weight, feat_set=v3_subset, BA=0.6318. Selected by BA ranking with SEN/SPE>=0.40 constraint and 0.01 tolerance rule.

**Selection rules applied:**
1. Filter strategies where mean SEN >= 0.40 AND mean SPE >= 0.40
2. Rank by BA mean (descending)
3. Tolerance: if top_BA - candidate_BA <= 0.01, prefer simpler strategy
   Simplicity order: none > class_weight > smote > smote_cw
