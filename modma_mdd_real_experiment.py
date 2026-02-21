import argparse
import pandas as pd
import numpy as np
import os
import glob
import logging
import warnings
import mne

logger = logging.getLogger(__name__)

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

def load_windows(participants_df, bids_root, window_sec, resample_sfreq, crop_duration=60.0):
    all_epochs = []
    labels = []
    groups = []
    
    label_map = {'MDD': 1, 'HC': 0}
    
    for _, row in participants_df.iterrows():
        sub_id = row['participant_id']
        group_label = row['group']
        
        safe_id = os.path.basename(sub_id)
        edf_pattern = os.path.join(bids_root, safe_id, 'eeg', f'{safe_id}_task-Resting-state_eeg.EDF')
        edf_files = glob.glob(edf_pattern)
        if not edf_files:
            edf_pattern = os.path.join(bids_root, safe_id, 'eeg', f'{safe_id}_task-Resting-state_eeg.edf')
            edf_files = glob.glob(edf_pattern)
            
        if not edf_files:
            logger.warning(f"找不到 {safe_id} 的 EDF 文件，跳过此人。")
            continue
            
        edf_path = edf_files[0]
        
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if raw.times[-1] > crop_duration:
                raw.crop(tmin=0, tmax=crop_duration)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
                if raw.info['sfreq'] != resample_sfreq:
                    raw.resample(resample_sfreq, verbose=False)
                    
            data = raw.get_data()
            window_size = int(window_sec * raw.info['sfreq'])
            
            for s in range(0, data.shape[1] - window_size + 1, window_size):
                segment = data[:, s:s + window_size]
                all_epochs.append(segment)
                labels.append(label_map[group_label])
                groups.append(sub_id)
                
        except Exception as e:
            logger.error(f"处理 {sub_id} 时发生错误: {e}")
            
    return np.array(all_epochs), np.array(labels), np.array(groups)

if __name__ == "__main__":
    args = parse_args()
    print(args)
