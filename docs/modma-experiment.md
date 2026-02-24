# MODMA MDD vs HC EEG Analysis

## Reproducible Pipeline Runbook

The enclosed script (`modma_mdd_real_experiment.py`) extracts power spectral density and frontal alpha asymmetry across independent EEG windows, performs group validation to isolate individual participants, and outputs objective classification significance with nested cross-validation and non-parametric permutation testing.

## Running the Verification

```bash
python modma_mdd_real_experiment.py \
    --bids-root "/path/to/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state" \
    --max-subjects 60 \
    --crop-duration 60 \
    --window-sec 10 \
    --resample-sfreq 125 \
    --min-windows-per-subject 3 \
    --n-permutations 1000 \
    --seed 42 \
    --output-dir results/modma_analysis
```

## Interpretation Rules

When reviewing `metrics.json` and associated outputs from the runs, apply the following rigorous guidelines:

* **Chance Level Performance**: If the subject-level Balanced Accuracy (BA) is near 0.5 (e.g. 50%) and the permutation `p >= 0.05`, the current feature set and baseline SVM *cannot reliably separate MDD from Healthy Controls (HC)* on this sample.
* **Overfitting / Variance Warnings**: If the BA shows improvement (e.g. > 0.60) but the 95% Bootstrap Confidence Interval (CI) is extremely wide or lower bound crosses 0.50, the model is unstable and not generalizing well. The recommendation is to increase the sample size `max_subjects` and stabilize features.
* **Clinical Implications**: *Never claim clinical usability or diagnostic capability from single-center retrospective cross-validation results.* Any high performance must be confirmed in an out-of-distribution independent cohort before practical application.
