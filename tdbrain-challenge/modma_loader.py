"""Load MODMA resting-state .mat files as MNE Raw objects."""
import numpy as np
import mne
from pathlib import Path
from scipy.io import loadmat

MODMA_RESTING_DIR = Path(
    r"D:\eeg\data\modma_resting\EEG_128channels_resting_lanzhou_2015"
)
SFREQ = 250.0  # MODMA default sampling rate


def get_modma_subjects():
    """Return list of (filepath, label_str) for all resting-state subjects.

    Label assignment by filename prefix:
      0201* -> MDD
      0202* / 0203* -> HC
    """
    mat_files = sorted(MODMA_RESTING_DIR.glob("*.mat"))
    subjects = []
    for f in mat_files:
        name = f.name
        if name.startswith("0201"):
            subjects.append((f, "MDD"))
        elif name.startswith("0202") or name.startswith("0203"):
            subjects.append((f, "HC"))
    return subjects


def _find_data_key(mat):
    """Dynamically discover the EEG data key in a loaded .mat dict.

    Skips dunder keys, 'samplingRate', and 'Impedances_*'.
    Returns the single remaining key whose value is a 2-D array.
    """
    candidates = [
        k for k in mat.keys()
        if not k.startswith("__")
        and k != "samplingRate"
        and not k.startswith("Impedances")
    ]
    # Filter to 2-D arrays with both dims > 1 (actual EEG data)
    data_keys = [
        k for k in candidates
        if hasattr(mat[k], "ndim") and mat[k].ndim == 2
        and min(mat[k].shape) > 1
    ]
    if len(data_keys) != 1:
        raise ValueError(
            f"Expected exactly 1 EEG data key, found {data_keys} "
            f"(all non-dunder keys: {candidates})"
        )
    return data_keys[0]


def load_modma_resting_raw(filepath):
    """Load a single MODMA .mat resting-state file as MNE RawArray.

    Dynamically discovers the data key (never hardcodes).
    Assumes data is in microvolts and converts to volts.

    Parameters
    ----------
    filepath : str or Path
        Path to the .mat file.

    Returns
    -------
    mne.io.RawArray
        EEG data with GSN-HydroCel-128 montage applied.
    """
    mat = loadmat(str(filepath))
    key = _find_data_key(mat)
    data = mat[key].astype(np.float64)  # (n_channels, n_samples) expected

    # Ensure shape is (n_channels, n_samples)
    if data.shape[0] > data.shape[1]:
        data = data.T

    n_ch = data.shape[0]

    # EGI HydroCel-128: channels E1-E128, row 129 is Cz reference
    ch_names = [f"E{i + 1}" for i in range(min(n_ch, 128))]
    data = data[: len(ch_names), :]

    # Read sampling rate from file if available, else use default
    sfreq = SFREQ
    if "samplingRate" in mat:
        sfreq = float(mat["samplingRate"].flat[0])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # uV -> V

    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    # Keep only channels present in the montage
    valid = [ch for ch in raw.ch_names if ch in montage.ch_names]
    raw.pick(valid)
    raw.set_montage(montage, verbose=False)
    return raw
