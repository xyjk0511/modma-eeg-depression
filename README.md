# MODMA EEG Depression Classification

MDD vs HC binary classification using 128-channel ERP data from the MODMA dataset. Statistically significant results achieved through QC repair, P300 feature extraction, electrode selection, time-window ablation, and feature importance analysis.

## Key Findings

| Model | Dims | BA | p-value | Significant |
|-------|------|----|---------|-------------|
| hcue P300 mean amplitude | 128 | 0.670 | 0.022 | Yes |
| Parietal subset (Pz/P3/P4/Cz/CPz) | 5 | 0.634 | 0.045 | Yes |
| 250-300ms time window | 128 | 0.673 | 0.018 | Yes |

- hcue dot-probe P300 mean amplitude is the only statistically significant MDD classification feature
- E55 (Cz-adjacent) has the highest channel weight, with positive deflection toward HC
- Feature enrichment and multi-condition fusion both degraded performance
- Cross-dataset generalization (TDBRAIN resting-state) failed (BA=0.500)

## Dataset

[MODMA](https://modma.lzu.edu.cn/data/application/) 128-channel ERP dataset:
- 53 subjects (24 MDD + 29 HC), 128-ch EGI HydroCel, 250Hz
- Dot-probe emotional attention bias paradigm (hcue/fcue/scue conditions)
- 52 subjects retained after QC (dynamic 12% bad-channel threshold + spherical spline interpolation)

## Methods

- **Preprocessing**: 1.0Hz highpass + 0.5-40Hz bandpass + pyprep PREP bad-channel detection
- **Features**: P300 mean amplitude (250-500ms window, 128 channels)
- **Classifier**: PCA(20) + LogisticRegression(C=1.0, balanced)
- **Validation**: Leave-One-Subject-Out CV + 1000-iteration permutation test

## Project Structure

```
scripts/modma/
  run_modma_erp.py              # ERP pipeline (P300/N200, classification, ablation)
  modma_mdd_real_experiment.py  # Resting-state pipeline (QC, band power)
  run_modma_real.py             # Resting-state classification entry
  run_modma_reproduce.py        # Reproduction script
  modma_mdd_classification.py   # Early classification experiments
tests/                          # Unit tests
docs/PRE_REGISTRATION.md        # Cross-dataset replication pre-registration
```

## Negative Results

| Attempt | Result | Conclusion |
|---------|--------|------------|
| P300+N200 enriched features (256-dim) | BA 0.670→0.554 | Curse of dimensionality |
| Three-condition fusion (384-dim) | BA 0.670→0.521 | Fusion introduces noise |
| Resting-state band power (640-dim) | BA=0.391 | Below chance level |
| TDBRAIN cross-dataset replication | BA=0.500 | Resting-state features don't generalize |

## Limitations

- N=52 limits generalizability
- No same-paradigm external validation dataset
- BA=0.670 is moderate; limited clinical applicability

## Usage

```bash
pip install mne scikit-learn pyprep numpy scipy joblib pandas matplotlib
python scripts/modma/run_modma_erp.py
```

Download data from [MODMA](https://modma.lzu.edu.cn/data/application/) into the `data/` directory first.

## License

MIT
