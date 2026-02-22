"""TDBRAIN external data adapter for confirmatory replication.

Loads TDBRAIN BrainVision (.vhdr) files, picks 10-20 channels, applies locked
preprocessing, returns features in the same format as the MODMA pipeline.
"""
import os
import glob
import logging
import warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

from modma_mdd_real_experiment import (
    extract_features,
    build_region_indices,
    STANDARD_1020_REGION_MAP,
    build_quality_mask,
)

logger = logging.getLogger(__name__)

VALID_1020_CHANNELS = set(STANDARD_1020_REGION_MAP.keys())


def load_tdbrain_subjects(tdbrain_root):
    """Parse participants.tsv, filter MDD + HC, return DataFrame.

    Raises ValueError if no HC subjects found (fallback needed).
    """
    tsv_path = os.path.join(tdbrain_root, "participants.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"participants.tsv not found at {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")

    # Normalize column names to lowercase for robustness
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify participant_id column
    id_col = None
    for candidate in ["participant_id", "subject_id", "sub"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(f"No participant ID column found. Columns: {list(df.columns)}")

    # Identify group/diagnosis column
    group_col = None
    for candidate in ["group", "diagnosis", "diag", "dx"]:
        if candidate in df.columns:
            group_col = candidate
            break
    if group_col is None:
        raise ValueError(f"No group/diagnosis column found. Columns: {list(df.columns)}")

    df[id_col] = df[id_col].astype(str).str.strip()
    df[group_col] = df[group_col].astype(str).str.strip()

    # Map common labels to MDD/HC
    label_map = {
        "MDD": "MDD", "mdd": "MDD", "depression": "MDD", "Depression": "MDD",
        "HC": "HC", "hc": "HC", "healthy": "HC", "Healthy": "HC",
        "control": "HC", "Control": "HC", "neurotypical": "HC",
    }
    df["group"] = df[group_col].map(label_map)
    df["participant_id"] = df[id_col]
    df = df.dropna(subset=["group"])
    df = df[df["group"].isin(["MDD", "HC"])][["participant_id", "group"]]

    n_mdd = (df["group"] == "MDD").sum()
    n_hc = (df["group"] == "HC").sum()
    logger.info(f"TDBRAIN subjects: {n_mdd} MDD, {n_hc} HC, {len(df)} total")

    if n_hc == 0:
        raise ValueError(
            "No HC subjects found in TDBRAIN. "
            "Fallback: use MODMA leave-N-out (exploratory only)."
        )
    return df.reset_index(drop=True)


def _spectrum_sanity_check(raw, sfreq_target):
    """One-time post-resample spectrum check on first subject.

    Logs alpha peak and checks for Nyquist aliasing artifacts.
    """
    # Pick one parietal/occipital channel for check
    check_chs = [c for c in raw.ch_names if c in ("O1", "O2", "Oz", "P3", "P4", "Pz")]
    if not check_chs:
        check_chs = raw.ch_names[:1]
    ch = check_chs[0]
    idx = raw.ch_names.index(ch)
    data = raw.get_data(picks=[idx])[0]

    freqs, psd = welch(data, fs=sfreq_target, nperseg=min(len(data), int(sfreq_target * 2)))

    # Alpha peak check (8-13 Hz)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    if alpha_mask.any():
        alpha_psd = psd[alpha_mask]
        peak_freq = freqs[alpha_mask][np.argmax(alpha_psd)]
        logger.info(f"Spectrum check ({ch}): alpha peak at {peak_freq:.1f} Hz")
        if peak_freq < 8 or peak_freq > 13:
            logger.warning(f"Alpha peak {peak_freq:.1f} Hz outside 8-13 Hz range")

    # Nyquist artifact check (near sfreq/2)
    nyquist = sfreq_target / 2.0
    high_mask = (freqs >= nyquist - 3) & (freqs <= nyquist)
    if high_mask.any():
        high_power = np.max(psd[high_mask])
        median_power = np.median(psd)
        if median_power > 0 and high_power / median_power > 10:
            logger.warning(
                f"Possible Nyquist artifact: power near {nyquist:.1f} Hz "
                f"is {high_power / median_power:.1f}x median"
            )
        else:
            logger.info(f"Spectrum check: no Nyquist artifact detected")


def load_tdbrain_features(
    tdbrain_root,
    resample_sfreq=125.0,
    crop_duration=60.0,
    highpass_freq=1.0,
    bad_amp_uv=200.0,
    window_sec=10.0,
    min_windows_per_subject=3,
):
    """Load TDBRAIN, preprocess with locked params, return features.

    Returns (features, y, groups, ch_names, qc_stats).
    """
    participants_df = load_tdbrain_subjects(tdbrain_root)
    label_map = {"MDD": 1, "HC": 0}
    thr_v = bad_amp_uv * 1e-6

    all_epochs, labels, groups = [], [], []
    ch_names_out = None
    spectrum_checked = False
    qc_stats = {"total_subjects": len(participants_df), "loaded": 0,
                "no_bdf": 0, "errors": 0, "windows_total": 0, "windows_kept": 0}

    for _, row in participants_df.iterrows():
        sub_id = row["participant_id"]
        group_label = row["group"]

        # Find BrainVision .vhdr file
        safe_id = os.path.basename(sub_id)
        vhdr_path = None
        # Search patterns: BIDS-like and flat layouts
        search_dirs = [
            os.path.join(tdbrain_root, safe_id, "eeg"),
            os.path.join(tdbrain_root, safe_id),
        ]
        for d in search_dirs:
            matches = glob.glob(os.path.join(d, "*.vhdr"))
            if matches:
                # Prefer eyes-closed resting state if multiple files
                ec = [m for m in matches if "EC" in os.path.basename(m)]
                vhdr_path = ec[0] if ec else matches[0]
                break

        if vhdr_path is None:
            logger.warning(f"No .vhdr file for {safe_id}, skipping")
            qc_stats["no_bdf"] += 1
            continue

        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

            # Pick only standard 10-20 channels
            keep_chs = [ch for ch in raw.ch_names if ch in VALID_1020_CHANNELS]
            if len(keep_chs) < 5:
                logger.warning(f"{safe_id}: only {len(keep_chs)} 10-20 channels, skipping")
                qc_stats["errors"] += 1
                continue
            raw.pick_channels(keep_chs)

            if ch_names_out is None:
                ch_names_out = raw.ch_names.copy()

            # Set montage
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="warn", verbose=False)

            # Crop
            if raw.times[-1] > crop_duration:
                raw.crop(tmin=0, tmax=crop_duration)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # Filter
                raw.filter(l_freq=highpass_freq, h_freq=45.0, verbose=False)

                # Interpolate bad channels (<=25% bad)
                data_raw = raw.get_data()
                bad_mask = np.any(np.abs(data_raw) > thr_v, axis=1)
                bad_chs = [raw.ch_names[i] for i in range(len(raw.ch_names)) if bad_mask[i]]
                if 0 < len(bad_chs) <= int(len(raw.ch_names) * 0.25):
                    raw.info["bads"] = bad_chs
                    raw.interpolate_bads(reset_bads=True, verbose=False)

                # Average reference
                raw.set_eeg_reference("average", verbose=False)

                # Resample
                if raw.info["sfreq"] != resample_sfreq:
                    raw.resample(resample_sfreq, verbose=False)

            # Post-resample spectrum sanity check (first subject only)
            if not spectrum_checked:
                _spectrum_sanity_check(raw, resample_sfreq)
                spectrum_checked = True

            # Window extraction (skip W0)
            data = raw.get_data()
            n_channels = data.shape[0]
            window_size = int(window_sec * resample_sfreq)
            max_bad_win = max(1, int(n_channels * 0.12))

            for s in range(window_size, data.shape[1] - window_size + 1, window_size):
                segment = data[:, s:s + window_size]
                qc_stats["windows_total"] += 1
                bad_ch_count = np.sum(np.any(np.abs(segment) > thr_v, axis=1))
                if bad_ch_count > max_bad_win:
                    continue
                all_epochs.append(segment)
                labels.append(label_map[group_label])
                groups.append(sub_id)
                qc_stats["windows_kept"] += 1

            qc_stats["loaded"] += 1

        except Exception as e:
            logger.error(f"Error processing {safe_id}: {e}")
            qc_stats["errors"] += 1

    if not all_epochs:
        raise ValueError("No valid windows loaded from TDBRAIN")

    X = np.array(all_epochs)
    y = np.array(labels)
    groups_arr = np.array(groups)

    # Validate min windows per subject
    unique_g, counts = np.unique(groups_arr, return_counts=True)
    valid_g = set(unique_g[counts >= min_windows_per_subject])
    mask = np.array([g in valid_g for g in groups_arr])
    X, y, groups_arr = X[mask], y[mask], groups_arr[mask]

    if len(X) == 0:
        raise ValueError("No subjects meet min_windows_per_subject threshold")

    # Extract features using locked pipeline
    features, feature_names = extract_features(X, sfreq=resample_sfreq, ch_names=ch_names_out)
    logger.info(f"TDBRAIN features: {features.shape}, names: {feature_names}")
    qc_stats["subjects_final"] = len(set(groups_arr))

    return features, y, groups_arr, ch_names_out, qc_stats


if __name__ == "__main__":
    """Self-test on synthetic dummy data (no real TDBRAIN needed)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Simulate 10-20 channel data
    ch_names = list(STANDARD_1020_REGION_MAP.keys())
    n_ch = len(ch_names)
    sfreq = 125.0
    duration = 60.0
    n_samples = int(sfreq * duration)

    rng = np.random.RandomState(42)
    # 3 subjects, 2 windows each (skip W0 -> need 30s+ of data)
    n_subjects = 6
    all_epochs = []
    labels_list = []
    groups_list = []
    window_size = int(10 * sfreq)

    for i in range(n_subjects):
        sub_id = f"sub-{i:03d}"
        label = 1 if i < 3 else 0  # 3 MDD, 3 HC
        # Generate 3 windows worth of data (skip W0 + keep 2)
        n_win = 3
        for w in range(1, n_win):  # skip W0
            segment = rng.randn(n_ch, window_size) * 10e-6  # ~10 uV
            all_epochs.append(segment)
            labels_list.append(label)
            groups_list.append(sub_id)

    X = np.array(all_epochs)
    y = np.array(labels_list)
    groups = np.array(groups_list)

    # Test extract_features
    features, names = extract_features(X, sfreq=sfreq, ch_names=ch_names)
    assert features.shape[0] == len(y), f"Feature rows {features.shape[0]} != labels {len(y)}"
    assert features.shape[1] == 5, f"Expected 5 features, got {features.shape[1]}"
    assert len(names) == 5, f"Expected 5 feature names, got {len(names)}"

    # Test build_region_indices with 10-20 channels
    regions = build_region_indices(ch_names)
    for r in ["frontal", "central", "temporal", "parietal", "occipital"]:
        assert len(regions[r]) > 0, f"Region {r} has no channels"

    # Test spectrum sanity check with synthetic raw
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw_dummy = mne.io.RawArray(rng.randn(n_ch, n_samples) * 10e-6, info, verbose=False)
    _spectrum_sanity_check(raw_dummy, sfreq)

    print(f"Self-test PASSED: {features.shape[1]} features, {len(names)} names, "
          f"{len(set(groups))} subjects, all regions mapped")
