# MDD-vs-Healthy Reframing Summary

## 1. Experiment Setup

- **Pipeline:** StandardScaler + SVC(kernel=rbf, class_weight=balanced, random_state=42)
- **Base features:** v3_subset (abs_bp + rel_bp + Hjorth, 676-dim, combined EO+EC)
- **CV:** 5-fold StratifiedGroupKFold
- **Statistical test:** scipy permutation_test, permutation_type=independent, n_resamples=9999

## 2. Dataset Composition

| Strategy | Total | MDD | nonMDD | Description |
|----------|-------|-----|--------|-------------|
| full_baseline | 1138 | 305 | 833 | All DISCOVERY subjects |
| mdd_vs_healthy | 350 | 305 | 45 | Exact MDD + exact HEALTHY only |
| exclude_high_fp | 904 | 305 | 599 | Remove 7 high-FP psychiatric groups |

## 3. Results Comparison

| Strategy | BA | AUC | SEN | SPE | delta-BA | p-value |
|----------|-----|-----|-----|-----|----------|--------|
| full_baseline | 0.6318 | 0.6884 | 0.6885 | 0.5750 | - | - |
| mdd_vs_healthy | 0.6488 | 0.6635 | 0.8754 | 0.4222 | +0.0170 | 0.317460 |
| exclude_high_fp | 0.6913 | 0.7434 | 0.8033 | 0.5793 | +0.0595 | 0.027778 |

## 4. Statistical Tests

- **mdd_vs_healthy vs full_baseline:** delta-BA=+0.0170, p=0.31746
- **exclude_high_fp vs full_baseline:** delta-BA=+0.0595, p=0.027778

## 5. Decision & Recommendation

- **Recommended strategy:** exclude_high_fp
- **Rationale:** exclude_high_fp significantly improves BA over full dataset (p<0.05). Psychiatric nonMDD confirmed as confound.
- **Adopt for v3.0:** True

## 6. Caveats

- mdd_vs_healthy has far fewer nonMDD subjects (45 HEALTHY vs 833 nonMDD), so higher BA may partly reflect an easier task rather than a better model
- exclude_high_fp is a more conservative approach that retains more subjects while removing known confounders
- The competition evaluation uses the full nonMDD pool, so filtered training may not generalize to the test set
