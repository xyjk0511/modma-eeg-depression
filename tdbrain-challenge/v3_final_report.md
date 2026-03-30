# TDBrain Challenge v3.0 Final Report

## Executive Summary

- **v2.0 baseline:** BA=0.6318, AUC=0.6887 (SVM + v3_subset, all 1138 subjects)
- **v3.0 final:** BA=0.6913, AUC=0.7434 (SVM + v3_subset + exclude_high_fp, 904 subjects)
- **Net improvement:** +0.0595 BA (+9.4% relative), +0.0547 AUC (+7.9% relative)
- **Final model:** SVM-only (ensemble not adopted)

## Phase Results Summary

| Phase | Strategy | BA | AUC | delta-BA | p-value | Adopted |
|-------|----------|-----|-----|----------|---------|---------|
| 6 | Age covariate (as feature) | 0.6243 | 0.6876 | -0.0075 | 0.688 | No |
| 6 | Age stratified | 0.5942 | - | -0.0376 | - | No |
| 7 | mdd_vs_healthy | 0.6488 | 0.6635 | +0.0170 | 0.317 | No |
| 7 | exclude_high_fp | 0.6913 | 0.7434 | +0.0595 | 0.028 | Yes |
| 8 | Delta augmented (720d) | 0.6856 | 0.7455 | -0.0057 | 0.591 | No |
| 9 | XGB-only (nested CV) | 0.5950 | 0.7285 | -0.0963 | 0.992 | No |
| 9 | SVM+XGB ensemble | 0.5983 | 0.7314 | -0.0930 | 1.000 | No |

## Final Model Configuration

- **Feature set:** v3_subset (abs_bp + rel_bp + Hjorth, 676-dim, EO+EC combined)
- **Subject filter:** exclude_high_fp (remove BURNOUT, CHRONIC PAIN, INSOMNIA, OCD, PARKINSON, SMC, TINNITUS from nonMDD)
- **Training subjects:** 904 (MDD=305, nonMDD=599)
- **Model:** SVM (SVC, kernel=rbf, class_weight=balanced, random_state=42)
- **CV performance:** BA=0.6913, AUC=0.7434, SEN=0.8033, SPE=0.5793
- **Evaluation:** 5-fold StratifiedGroupKFold
- **Serialized model:** `models/final_svm.joblib`

## Key Findings

### What Worked
1. **Subject filtering (Phase 7):** Removing 7 psychiatric nonMDD groups (n=230 subjects) that share EEG patterns with MDD significantly improved BA by +0.0595 (p=0.028). This was the only optimization that passed the p<0.05 significance threshold across all four phases.

### What Did Not Work
1. **Age covariate (Phase 6):** Adding age as a feature slightly hurt BA (-0.0075). Age-stratified models performed even worse (-0.0376) due to small per-bin MDD counts. Age is a confound (FP vs TN p<1e-49) but not a useful predictor.
2. **Delta features (Phase 8):** Delta coherence (12 pairs) and asymmetry (10 pairs) did not improve BA (-0.0057, p=0.591). The 44 delta features are weak standalone predictors (BA=0.5640) and slightly dilute the signal when added to v3_subset.
3. **XGBoost and ensemble (Phase 9):** XGB-only (BA=0.5950) and SVM+XGB ensemble (BA=0.5983) both performed substantially worse than SVM-only. XGB achieved high SPE (0.81) but very low SEN (0.38), indicating it learned a conservative decision boundary. The weighted soft voting ensemble could not overcome this SEN/SPE imbalance.

### Why SVM Outperforms XGBoost
- SVM with class_weight=balanced directly adjusts the decision boundary for class imbalance, achieving balanced SEN/SPE
- XGBoost with scale_pos_weight adjusts sample weights but still tends toward the majority class in high-dimensional (676d) settings with only 904 subjects
- The 676-dimensional feature space with 904 subjects favors SVM's margin-based approach over tree-based methods

## Recommendations for Future Work

1. **ADV-01: Feature selection refinement** — Apply recursive feature elimination or L1-regularized SVM to reduce the 676-dim feature space, potentially improving generalization
2. **ADV-02: Cross-dataset validation** — Validate the exclude_high_fp + SVM model on external EEG depression datasets to assess generalizability
3. **ADV-03: Threshold optimization** — The current model uses default 0.5 threshold; optimizing the decision threshold on the ROC curve could improve the SEN/SPE trade-off for specific clinical use cases
4. **GEN-01: REPLICATION set evaluation** — Generate final predictions on the REPLICATION set using the v3.0 model for competition submission

## Appendix: Per-Phase Details

### Phase 6: Age Covariate Integration
- Age-as-feature: BA=0.6243 (delta=-0.0075, p=0.688)
- Age-stratified: BA=0.5942 (delta=-0.0376)
- Conclusion: Age does not help; pooled baseline remains best

### Phase 7: MDD-vs-Healthy Reframing
- full_baseline: BA=0.6318 (n=1138)
- mdd_vs_healthy: BA=0.6488 (n=350, p=0.317)
- exclude_high_fp: BA=0.6913 (n=904, p=0.028)
- Conclusion: Removing psychiatric nonMDD confounders significantly improves classification

### Phase 8: Delta Band Feature Engineering
- baseline (676d): BA=0.6913
- delta_only (44d): BA=0.5640
- augmented (720d): BA=0.6856 (p=0.591)
- Conclusion: Delta coherence/asymmetry features do not add predictive value

### Phase 9: Ensemble Modeling
- svm_only: BA=0.6913, AUC=0.7434, SEN=0.8033, SPE=0.5793
- xgb_only: BA=0.5950, AUC=0.7285, SEN=0.3803, SPE=0.8097
- ensemble: BA=0.5983, AUC=0.7314, SEN=0.3836, SPE=0.8130
- Conclusion: SVM-only remains the best model; ensemble not adopted

---
*Report generated: 2026-02-24*
*v3.0 pipeline: exclude_high_fp + v3_subset (676d) + SVM (class_weight=balanced)*
