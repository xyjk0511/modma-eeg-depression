import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

import traceback
from modma_mdd_real_experiment import run_main_with_output_dir

try:
    run_main_with_output_dir(
        bids_root=str(ROOT / "data/modma_bids/EEG_LZU_2015_2_resting state"),
        output_dir=str(ROOT / "outputs/results/modma_run_01"),
        max_subjects=40,
        resample_sfreq=125.0,
        n_permutations=100,
        seed=42,
        min_windows_per_subject=3,
        window_sec=10.0,
        crop_duration=60.0
    )
except Exception as e:
    with open(str(ROOT / "err_real2.txt"), "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
