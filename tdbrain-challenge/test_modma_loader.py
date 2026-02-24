"""Tests for MODMA resting-state data loader."""
import numpy as np
import pytest
from modma_loader import load_modma_resting_raw, get_modma_subjects


def test_get_modma_subjects_returns_list():
    """Should return a non-empty list of (filepath, label) tuples."""
    subjects = get_modma_subjects()
    assert len(subjects) > 0, "No subjects found"
    for fpath, label in subjects:
        assert label in ("MDD", "HC"), f"Unexpected label: {label}"
        assert fpath.exists(), f"File not found: {fpath}"


def test_get_modma_subjects_has_both_classes():
    """Both MDD and HC labels should be present."""
    subjects = get_modma_subjects()
    labels = {label for _, label in subjects}
    assert "MDD" in labels, "No MDD subjects found"
    assert "HC" in labels, "No HC subjects found"


def test_load_modma_resting_raw_returns_rawarray():
    """Loaded object should be an MNE RawArray."""
    subjects = get_modma_subjects()
    raw = load_modma_resting_raw(subjects[0][0])
    assert isinstance(raw, mne.io.RawArray)


def test_load_modma_resting_raw_channel_count():
    """Should have 128 EGI channels after montage filtering."""
    subjects = get_modma_subjects()
    raw = load_modma_resting_raw(subjects[0][0])
    # GSN-HydroCel-128 montage has E1-E128
    assert raw.info["nchan"] == 128, f"Expected 128 channels, got {raw.info['nchan']}"


def test_load_modma_resting_raw_sfreq():
    """Sampling rate should be 250 Hz."""
    subjects = get_modma_subjects()
    raw = load_modma_resting_raw(subjects[0][0])
    assert raw.info["sfreq"] == 250.0


def test_load_modma_resting_raw_has_montage():
    """Raw should have a montage set (channel positions available)."""
    subjects = get_modma_subjects()
    raw = load_modma_resting_raw(subjects[0][0])
    montage = raw.get_montage()
    assert montage is not None, "Montage not set"


import mne  # noqa: E402 (needed for isinstance check above)
