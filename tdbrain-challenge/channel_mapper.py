"""Map EGI HydroCel-128 channels to standard 10-20 (TDBrain 26ch)."""
import numpy as np
import mne

TDBRAIN_CHANNELS = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'FC3','FCz','FC4','T7','C3','Cz','C4','T8',
    'CP3','CPz','CP4','P7','P3','Pz','P4','P8',
    'O1','Oz','O2',
]


def map_egi128_to_1020(egi_ch_names):
    """Find nearest EGI-128 electrode for each TDBrain 10-20 channel via 3D coords.

    Returns: dict {egi_channel_name: standard_1020_name}
    """
    montage_egi = mne.channels.make_standard_montage("GSN-HydroCel-128")
    montage_1020 = mne.channels.make_standard_montage("standard_1020")

    egi_pos = montage_egi.get_positions()["ch_pos"]
    std_pos = montage_1020.get_positions()["ch_pos"]

    mapping = {}
    used_egi = set()
    for std_ch in TDBRAIN_CHANNELS:
        if std_ch not in std_pos:
            continue
        target = std_pos[std_ch]
        best_egi, best_dist = None, np.inf
        for egi_ch in egi_ch_names:
            if egi_ch in used_egi or egi_ch not in egi_pos:
                continue
            dist = np.linalg.norm(egi_pos[egi_ch] - target)
            if dist < best_dist:
                best_dist = dist
                best_egi = egi_ch
        if best_egi is not None:
            mapping[best_egi] = std_ch
            used_egi.add(best_egi)

    return mapping
