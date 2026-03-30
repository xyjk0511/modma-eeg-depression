# TDBrain Challenge v2.0 Diagnostic Report

**Generated:** 2026-02-24
**Project:** EEG-based MDD classification
**Dataset:** TDBrain (1138 subjects, 305 MDD, 833 nonMDD)

## Executive Summary

### Key Achievements
- **Best configuration:** v3_subset (abs_bp + rel_bp + Hjorth, 676-dim) + class_weight=balanced
- **Performance:** BA=0.6318, AUC=0.6887 (SEN=0.689, SPE=0.575)
- **Improvement over baseline:** +0.0667 BA (+11.8% relative)
- **Statistical significance:** p=0.001, Cohen's d=6.28 (large effect)

### Top 3 Recommendations for v3.0
1. **Add age as model covariate** (expected: 0.02-0.05 BA, ROI=1.50)
2. **MDD-vs-healthy only (exclude psychiatric nonMDD)** (expected: 0.03-0.05 BA, ROI=1.50)
3. **Focus on delta band features** (expected: 0.005-0.015 BA, ROI=1.00)

### Critical Findings
- Age is the dominant confound: FP vs TN mean age 45.9 vs 25.7 years (p=6.59e-49)
- Entropy features add noise: removing them improves BA by +0.0537
- SVM without class rebalancing is degenerate: SEN=0.013 (predicts all nonMDD)

## Performance Evolution

| Phase | Configuration | BA | AUC | Notes |
|-------|--------------|-----|-----|-------|
| Baseline | All features (992-dim), no SelectKBest | 0.5651 | 0.6535 | Starting point |
| Phase 1 | abs_bp only (260-dim) | 0.6178 | 0.6587 | Best single category |
| Phase 2 | v3_subset + class_weight=balanced (0.6318 BA) | 0.6318 | 0.6887 | Best imbalance strategy |
| Phase 3 | v3_subset (statistically validated) | 0.6318 | --- | p=0.001, Cohen's d=6.28 |

## Phase 1: Feature Ablation

### Key Findings
- **Best single category:** abs_bp (BA=0.6178, AUC=0.6587, 260-dim)
- **Best sub-band:** abs_bp_delta (BA=0.636, AUC=0.6812, 52-dim)
- **Entropy removal:** LOO BA improved from 0.5651 to 0.6188 (+0.0537)
- **Recommended subset:** abs_bp + rel_bp + Hjorth (drop entropy) = v3_subset (676-dim)

### Single-Category Comparison (combined, SVM)

| Category | BA | AUC | Dims |
|----------|-----|-----|------|
| abs_bp | 0.6178 | 0.6587 | 260 |
| rel_bp | 0.6015 | 0.6688 | 260 |
| Hjorth | 0.5935 | 0.6502 | 156 |
| TBR_FAA | 0.5899 | 0.6098 | 4 |
| entropy | 0.5513 | 0.6256 | 312 |

## Phase 2: Class Imbalance Strategies

### Key Findings
- **Winner:** class_weight=balanced on v3_subset (BA=0.6318, SEN=0.689, SPE=0.575)
- **None strategy failure:** SEN=0.013 -- SVM without rebalancing predicts almost all nonMDD
- **SMOTE == SMOTE+class_weight:** double-compensation has no measurable effect

### Strategy Comparison

| Strategy | Feature Set | BA | AUC | SEN | SPE |
|----------|------------|-----|-----|-----|-----|
| none | v3_subset | 0.5024 | 0.6510 | 0.0131 | 0.9916 |
| none | full | 0.4958 | 0.6584 | 0.0000 | 0.9916 |
| class_weight | v3_subset | 0.6318 | 0.6887 | 0.6885 | 0.5750 |
| class_weight | full | 0.6147 | 0.6746 | 0.6459 | 0.5834 |
| smote | v3_subset | 0.6155 | 0.6744 | 0.5913 | 0.6399 |
| smote | full | 0.5984 | 0.6661 | 0.5170 | 0.6799 |
| smote_cw | v3_subset | 0.6155 | 0.6744 | 0.5913 | 0.6399 |
| smote_cw | full | 0.5984 | 0.6661 | 0.5170 | 0.6799 |

## Phase 3: Statistical Validation

### Key Findings
- **Both feature sets significant:** v3_subset p=0.001, full p=0.001 (Bonferroni threshold=0.025)
- **v3_subset effect size:** Cohen's d=6.28 (large)
- **full effect size:** Cohen's d=5.50 (large)
- **Null distribution:** mean=0.5007, std=0.0209, 95% CI=[0.4573, 0.5408]
- **Permutations:** 1000

### Permutation Test Results

| Feature Set | Observed BA | Null Mean | Cohen's d | p-value | Significant |
|-------------|------------|-----------|-----------|---------|-------------|
| v3_subset | 0.6318 | 0.5007 | 6.28 | 0.001 | Yes |
| full | 0.6147 | 0.5012 | 5.50 | 0.001 | Yes |

## Phase 4: Error Analysis

### Key Findings
- **Optimal threshold:** Youden's J = 0.2469 (vs default 0.5)
- **Confusion matrix:** TP=230, TN=440, FP=393, FN=75
- **Age confound:** correct mean age 32.8 vs wrong 45.6 years (p=2.21e-27)
- **Gender bias:** p=0.004
- **Education bias:** p=5.17e-22
- **Feature-space distance:** NOT significant (p=0.270) -- errors driven by demographics, not feature outliers

### FP Rate by nonMDD Indication

| Indication | n | FP | FP Rate |
|------------|---|----|---------| 
| INSOMNIA * | 30 | 27 | 0.9000 |
| SMC * | 86 | 57 | 0.6628 |
| TINNITUS * | 29 | 19 | 0.6552 |
| OCD * | 37 | 24 | 0.6486 |
| PARKINSON * | 25 | 16 | 0.6400 |
| CHRONIC PAIN * | 13 | 8 | 0.6154 |
| BURNOUT * | 10 | 5 | 0.5000 |
| HEALTHY | 45 | 20 | 0.4444 |
| ADHD | 188 | 59 | 0.3138 |
| Dyslexia | 17 | 0 | 0.0000 |

*\* Above overall FP rate (0.4896)*

## v3.0 Optimization Roadmap

Prioritized by ROI (expected gain / implementation cost):

### 1. Add age as model covariate

- **Category:** Covariate Modeling
- **Source:** Phase 4
- **Evidence:** FP vs TN age gap: mean 45.9 vs 25.7 years (p<1e-49)
- **Expected gain:** 0.02-0.05 BA
- **Effort:** Medium
- **Risk:** Low
- **ROI score:** 1.50

**Implementation approach:**
1. Add age as linear feature alongside EEG features
2. Alternatively: age-stratified models (young/middle/old)
3. Validate with nested CV to avoid overfitting

### 2. MDD-vs-healthy only (exclude psychiatric nonMDD)

- **Category:** Problem Reframing
- **Source:** Phase 4
- **Evidence:** 7 nonMDD subgroups with FP rate > 0.49; INSOMNIA FP=0.90
- **Expected gain:** 0.03-0.05 BA
- **Effort:** Low
- **Risk:** Medium
- **ROI score:** 1.50

**Implementation approach:**
1. Filter dataset to MDD + HEALTHY subjects only
2. Re-run SVM pipeline with class_weight=balanced
3. Compare BA to full-dataset result to quantify confound impact

### 3. Focus on delta band features

- **Category:** Feature Engineering
- **Source:** Phase 1
- **Evidence:** abs_bp_delta: BA=0.636 (best sub-band, 52-dim)
- **Expected gain:** 0.005-0.015 BA
- **Effort:** Low
- **Risk:** Low
- **ROI score:** 1.00

**Implementation approach:**
1. Extract additional delta-band features (coherence, asymmetry)
2. Run ablation to confirm incremental value
3. Integrate into v3_subset if BA gain > 0.01

### 4. Subgroup-specific decision thresholds

- **Category:** Advanced Modeling
- **Source:** Phase 4
- **Evidence:** Youden's J optimal threshold=0.2469 (vs default 0.5); age-dependent FP rates
- **Expected gain:** 0.01-0.03 BA
- **Effort:** Medium
- **Risk:** Low
- **ROI score:** 1.00

**Implementation approach:**
1. Compute age-bin-specific optimal thresholds via Youden's J
2. Implement threshold lookup at prediction time
3. Validate with LOSO CV to avoid overfitting thresholds

### 5. Ensemble methods (SVM + XGB voting)

- **Category:** Advanced Modeling
- **Source:** Cross-phase
- **Evidence:** SVM BA=0.6318 (high SEN), XGB shows complementary SPE patterns in Phase 1
- **Expected gain:** 0.01-0.03 BA
- **Effort:** Medium
- **Risk:** Low
- **ROI score:** 1.00

**Implementation approach:**
1. Train SVM and XGB with class_weight=balanced on v3_subset
2. Combine via soft voting (average probabilities)
3. Validate with 5-fold CV, compare to SVM-only baseline

### 6. Multi-class classification (MDD vs specific conditions)

- **Category:** Problem Reframing
- **Source:** Phase 4
- **Evidence:** 7 nonMDD subgroups with above-average FP rates
- **Expected gain:** 0.05-0.10 BA (for MDD-vs-healthy)
- **Effort:** High
- **Risk:** High
- **ROI score:** 0.44

**Implementation approach:**
1. Define class hierarchy: MDD / psychiatric-nonMDD / healthy
2. Train one-vs-rest SVM ensemble
3. Evaluate per-class metrics and overall BA

### 7. Gender-specific feature selection or models

- **Category:** Covariate Modeling
- **Source:** Phase 4
- **Evidence:** Gender bias in misclassification (p=0.0042)
- **Expected gain:** 0.005-0.02 BA
- **Effort:** Medium
- **Risk:** Medium
- **ROI score:** 0.38

**Implementation approach:**
1. Split dataset by gender, train separate SVM models
2. Compare per-gender BA to pooled model
3. If improvement > 0.01 BA, adopt gender-stratified approach

### 8. Cross-dataset validation with external EEG-MDD data

- **Category:** Data Augmentation
- **Source:** Phase 3
- **Evidence:** Current validation on single TDBrain dataset (0.6318 BA, n=1138)
- **Expected gain:** 0.00-0.02 BA (generalizability, not raw gain)
- **Effort:** High
- **Risk:** Medium
- **ROI score:** 0.17

**Implementation approach:**
1. Identify compatible EEG-MDD datasets (e.g., MODMA, MPI-Leipzig)
2. Harmonize feature extraction pipelines
3. Train on TDBrain, test on external; report transfer BA

## Lessons Learned

### Negative Results
- **Entropy adds noise:** Single-category entropy BA=0.5513 (worst category); removing entropy via LOO improves BA by +0.0537
- **SMOTE underperforms class_weight:** SMOTE BA=0.6155 vs class_weight BA=0.6318 on v3_subset
- **No-resampling SVM is degenerate:** SEN=0.013, predicts almost all nonMDD
- **SMOTE + class_weight = SMOTE alone:** double-compensation has zero measurable effect
- **Feature-space distance not predictive:** p=0.270, misclassification is demographic, not feature-driven

## Limitations and Risks

### Dataset Limitations
- Single dataset (TDBrain, n=1138): generalizability unknown
- Class imbalance (305 MDD vs 833 nonMDD, ratio 1:2.7)
- Heterogeneous nonMDD group includes psychiatric conditions that may share EEG features with MDD

### Methodological Limitations
- 3-fold CV used for primary evaluation (5-fold for revalidation only)
- SVM with fixed hyperparameters (C=1, RBF kernel) -- no hyperparameter tuning performed
- Platt scaling for probability calibration may be suboptimal for imbalanced data

### Risks for v3.0
- Expected gains are per-optimization, NOT cumulative (interaction effects likely)
- Age covariate modeling risks overfitting on small subgroups
- Problem reframing (MDD-vs-healthy) changes the clinical question and may not be acceptable

## Conclusion

The v2.0 diagnostic pipeline achieves BA=0.6318 (AUC=0.6887) using an SVM classifier with class_weight=balanced on the v3_subset feature set (abs_bp + rel_bp + Hjorth, 676 dimensions). This represents a +0.0667 BA improvement over the all-features baseline (0.5651), and is statistically significant at p=0.001 with a large effect size (Cohen's d=6.28).

The most actionable finding for v3.0 is the strong age confound: older nonMDD subjects are systematically misclassified as MDD, suggesting that age-related EEG changes overlap with MDD signatures. Adding age as a covariate or stratifying by age group is the highest-ROI optimization. Delta-band feature focus and MDD-vs-healthy problem reframing are complementary low-effort improvements.

Key negative results -- entropy features adding noise, SMOTE underperforming class_weight, and feature-space distance being non-predictive of errors -- provide clear guardrails for v3.0 development. The recommended approach is sequential implementation of top-3 ROI opportunities with validation after each step, rather than simultaneous changes.

## Appendix: Methodology

### Phase 1: Feature Ablation
- Coarse ablation: single-category and leave-one-out strategies across 5 feature categories (abs_bp, rel_bp, Hjorth, entropy, TBR_FAA) x 3 conditions (EO, EC, combined) x 2 models (SVM, XGB)
- Fine-grained ablation: per-frequency-band analysis of top 3 categories (delta, theta, alpha, beta, gamma)
- 5-fold revalidation of top 5 configurations

### Phase 2: Class Imbalance Strategies
- 4 strategies: none, class_weight=balanced, SMOTE, SMOTE+class_weight
- 2 feature sets: v3_subset (676-dim), full (992-dim)
- 3 random seeds for robustness, 5-fold CV per seed
- Selection: BA ranking with SEN/SPE >= 0.40 constraint

### Phase 3: Statistical Validation
- 1000-iteration permutation test with group-level label shuffling
- Bonferroni correction for 2 comparisons (threshold p<0.025)
- Phipson-Smyth corrected p-values
- Cohen's d effect size relative to null distribution

### Phase 4: Error Analysis
- Out-of-fold (OOF) probability collection for all 1138 subjects
- Youden's J optimal threshold (0.2469) for TP/TN/FP/FN categorization
- Mann-Whitney U tests for continuous covariates (age, education)
- Chi-squared tests for categorical covariates (gender)
- Bonferroni correction across 7 tests (threshold p<0.0071)
- FP rate analysis by nonMDD indication subgroup (n >= 10 filter)
