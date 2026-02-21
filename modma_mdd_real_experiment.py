import argparse
import pandas as pd
import numpy as np

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MODMA MDD vs HC Classification")
    parser.add_argument("--bids-root", required=True, help="Path to MODMA BIDS dataset")
    parser.add_argument("--max-subjects", type=int, default=None, help="Max subjects to process")
    parser.add_argument("--resample-sfreq", type=float, default=125.0, help="Resampling frequency")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-windows-per-subject", type=int, default=3, help="Min windows per subject")
    
    return parser.parse_args(argv)

def load_participants(path):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    return df[df["group"].isin(["MDD", "HC"])][["participant_id", "group"]]

def build_quality_mask(X, bad_amp_uv=200.0, max_bad_channels=3):
    thr_v = bad_amp_uv * 1e-6
    bad_ch = np.any(np.abs(X) > thr_v, axis=2)
    bad_ch_count = bad_ch.sum(axis=1)
    return bad_ch_count <= max_bad_channels

def validate_post_qc_availability(groups, y, keep_mask, min_windows_per_subject=3):
    kept_groups = groups[keep_mask]
    unique_groups, counts = np.unique(kept_groups, return_counts=True)
    for g, c in zip(unique_groups, counts):
        if c < min_windows_per_subject:
            raise ValueError(f"Subject {g} has only {c} windows after QC (min is {min_windows_per_subject} min_windows_per_subject)")

def validate_subject_class_counts(subject_labels):
    classes, counts = np.unique(list(subject_labels.values()), return_counts=True)
    for c, count in zip(classes, counts):
        if count < 2:
            raise ValueError(f"Class {c} has only {count} subjects after QC (need at least 2 subjects per class)")

if __name__ == "__main__":
    args = parse_args()
    print(args)
