# Age Covariate Integration Summary

## 1. Experiment Setup

- **Pipeline:** StandardScaler + SVC(kernel=rbf, class_weight=balanced, random_state=42)
- **Base features:** v3_subset (abs_bp + rel_bp + Hjorth, 676-dim, combined EO+EC)
- **CV:** 5-fold StratifiedGroupKFold
- **Age tertiles:** young <= 26.7, middle <= 49.1, old > 49.1
- **Subjects with valid age:** 1122

## 2. Results Comparison

| Strategy | BA | AUC | SEN | SPE | delta-BA | p-value |
|----------|-----|-----|-----|-----|----------|--------|
| Pooled Baseline | 0.6318 | 0.6884 | 0.6885 | 0.5750 | - | - |
| Age-as-Feature | 0.6243 | 0.6876 | 0.6623 | 0.5863 | -0.0075 | 0.687500 |
| Age-Stratified | 0.5942 | 0.6957 | 0.4295 | 0.7589 | -0.0376 | 0.968750 |
| Age-Interaction | 0.6247 | 0.6892 | 0.6656 | 0.5838 | -0.0071 | 0.656250 |

## 3. Age-Stratified Breakdown

| Age Bin | n | BA | AUC | SEN | SPE |
|---------|---|-----|-----|-----|-----|
| young | 374 | 0.6420 | 0.7927 | 0.4286 | 0.8555 |
| middle | 375 | 0.5662 | 0.5494 | 0.4126 | 0.7198 |
| old | 373 | 0.5557 | 0.5922 | 0.4488 | 0.6626 |

## 4. Best Strategy

- **Name:** baseline
- **BA:** 0.6318
- **Delta-BA over baseline:** +0.0000
- **Statistically significant (p<0.05):** False

## 5. Conclusion

No age integration strategy provides a statistically significant improvement over the pooled baseline (BA=0.6318). The best strategy (baseline) achieved BA=0.6318 (delta=+0.0000, p=1.0).
