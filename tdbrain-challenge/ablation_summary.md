# Feature Ablation Study Summary
## 1. Coarse Results (SVM, Combined)
Baseline: BA=0.5651, AUC=0.6535
| Category | Dims | Single BA | Single AUC | LOO BA | LOO AUC |
|----------|------|-----------|------------|--------|--------|
| abs_bp | 260 | 0.6178 | 0.6587 | 0.5619 | 0.6468 |
| rel_bp | 260 | 0.6015 | 0.6688 | 0.5678 | 0.6355 |
| Hjorth | 156 | 0.5935 | 0.6502 | 0.5704 | 0.6533 |
| TBR_FAA | 4 | 0.5899 | 0.6098 | 0.5695 | 0.6533 |
| entropy | 312 | 0.5513 | 0.6256 | 0.6188 | 0.6757 |

## 2. Coarse Results (XGB, Combined)
Baseline: BA=0.5703, AUC=0.6793
| Category | Dims | Single BA | Single AUC | LOO BA | LOO AUC |
|----------|------|-----------|------------|--------|--------|
| abs_bp | 260 | 0.5818 | 0.6569 | 0.5431 | 0.6340 |
| Hjorth | 156 | 0.5786 | 0.6673 | 0.5594 | 0.6624 |
| rel_bp | 260 | 0.5718 | 0.6725 | 0.5470 | 0.6475 |
| TBR_FAA | 4 | 0.5432 | 0.5715 | 0.5533 | 0.6604 |
| entropy | 312 | 0.5396 | 0.6127 | 0.5588 | 0.6666 |

## 3. Fine-grained Per-band Results (Combined)
| Sub-group | Model | Dims | BA | AUC |
|-----------|-------|------|----|-----|
| Hjorth_activity | svm | 52 | 0.5956 | 0.6517 |
| Hjorth_activity | xgb | 52 | 0.5697 | 0.6422 |
| Hjorth_complexity | svm | 52 | 0.5969 | 0.6528 |
| Hjorth_complexity | xgb | 52 | 0.5680 | 0.6331 |
| Hjorth_mobility | svm | 52 | 0.5994 | 0.6461 |
| Hjorth_mobility | xgb | 52 | 0.5717 | 0.6379 |
| abs_bp_alpha | svm | 52 | 0.5776 | 0.6107 |
| abs_bp_alpha | xgb | 52 | 0.5289 | 0.5771 |
| abs_bp_beta | svm | 52 | 0.5677 | 0.5862 |
| abs_bp_beta | xgb | 52 | 0.5602 | 0.5834 |
| abs_bp_delta | svm | 52 | 0.6360 | 0.6812 |
| abs_bp_delta | xgb | 52 | 0.5812 | 0.6629 |
| abs_bp_gamma | svm | 52 | 0.5424 | 0.5565 |
| abs_bp_gamma | xgb | 52 | 0.5057 | 0.5212 |
| abs_bp_theta | svm | 52 | 0.6052 | 0.6347 |
| abs_bp_theta | xgb | 52 | 0.5874 | 0.6477 |
| rel_bp_alpha | svm | 52 | 0.5941 | 0.6384 |
| rel_bp_alpha | xgb | 52 | 0.5542 | 0.6054 |
| rel_bp_beta | svm | 52 | 0.5853 | 0.6404 |
| rel_bp_beta | xgb | 52 | 0.5773 | 0.6447 |
| rel_bp_delta | svm | 52 | 0.6207 | 0.6721 |
| rel_bp_delta | xgb | 52 | 0.5916 | 0.6525 |
| rel_bp_gamma | svm | 52 | 0.5620 | 0.6178 |
| rel_bp_gamma | xgb | 52 | 0.5682 | 0.6211 |
| rel_bp_theta | svm | 52 | 0.5431 | 0.5785 |
| rel_bp_theta | xgb | 52 | 0.5123 | 0.5623 |

## 4. 5-fold Revalidation
| Original Key | Model | 3-fold BA | 5-fold BA | 5-fold AUC |
|-------------|-------|-----------|-----------|------------|
| fine_abs_bp_delta_combined_svm | svm | 0.6360 | 0.6271 | 0.6837 |
| fine_abs_bp_theta_combined_svm | svm | 0.6052 | 0.6089 | 0.6317 |
| fine_rel_bp_delta_combined_svm | svm | 0.6207 | 0.6199 | 0.6644 |
| single_category_abs_bp_combined_svm | svm | 0.6178 | 0.6210 | 0.6617 |
| single_category_rel_bp_combined_svm | svm | 0.6015 | 0.5951 | 0.6644 |

## 5. Conclusions
**Ablation baseline (no SelectKBest):** SVM BA=0.5651, AUC=0.6535

**External reference (main_diagnostic with SelectKBest):** SVM AUC=0.668, BA=0.613; SelectKBest k=200 BA=0.627

**Category contribution ranking (single-category BA, combined SVM):**
1. abs_bp: BA=0.6178
2. rel_bp: BA=0.6015
3. Hjorth: BA=0.5935
4. TBR_FAA: BA=0.5899
5. entropy: BA=0.5513

**Key findings:**
- abs_bp is the strongest single category (BA=0.6178)
- Removing entropy improves LOO BA above baseline, suggesting it adds noise
- TBR_FAA contributes modestly despite only 4 dims
- Recommended v3.0 feature subset: abs_bp + rel_bp + Hjorth (drop entropy, keep TBR_FAA optional)
