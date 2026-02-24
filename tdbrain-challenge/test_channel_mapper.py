import numpy as np
import mne
from channel_mapper import map_egi128_to_1020, TDBRAIN_CHANNELS


def test_map_returns_correct_channel_count():
    """Mapped output should have exactly 26 channels matching TDBrain."""
    montage_egi = mne.channels.make_standard_montage("GSN-HydroCel-128")
    egi_names = montage_egi.ch_names[:128]
    mapping = map_egi128_to_1020(egi_names)
    assert len(mapping) == 26
    assert set(mapping.values()) == set(TDBRAIN_CHANNELS)


def test_map_known_electrodes():
    """Verify known EGI-128 to 10-20 correspondences."""
    montage_egi = mne.channels.make_standard_montage("GSN-HydroCel-128")
    egi_names = montage_egi.ch_names[:128]
    mapping = map_egi128_to_1020(egi_names)
    for egi_ch, std_ch in mapping.items():
        assert egi_ch in egi_names
        assert std_ch in TDBRAIN_CHANNELS
