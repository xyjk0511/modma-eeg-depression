import logging
import numpy as np
import pandas as pd
import pytest
import mne as real_mne
from types import SimpleNamespace

import run_modma_real as mod


def test_run_evaluation_raises_on_subject_level_single_class():
    X = np.random.randn(20, 4)
    y = np.zeros(20, dtype=int)
    groups = np.array([f"sub-{i:03d}" for i in range(20)])

    with pytest.raises(ValueError, match="受试者级标签仅有单一类别"):
        mod.run_evaluation(X, y, groups)


def test_run_evaluation_raises_on_insufficient_subjects_per_class():
    # 3 个 0 类受试者 + 1 个 1 类受试者，至少一个类别受试者不足 2
    groups = np.repeat(["g1", "g2", "g3", "g4"], 5)
    y = np.array([0] * 15 + [1] * 5)
    X = np.random.randn(len(y), 4)

    with pytest.raises(ValueError, match="独立受试者少于 2 人"):
        mod.run_evaluation(X, y, groups)


def test_load_real_data_does_not_log_zero_loaded_progress(monkeypatch, caplog):
    df = pd.DataFrame(
        {
            "participant_id": ["sub-001", "sub-025"],
            "group": ["MDD", "HC"],
        }
    )

    monkeypatch.setattr(mod.os.path, "exists", lambda _: True)
    monkeypatch.setattr(mod.pd, "read_csv", lambda *args, **kwargs: df.copy())
    monkeypatch.setattr(mod.glob, "glob", lambda pattern: ["fake.edf"])

    class FakeRaw:
        def __init__(self):
            self.times = np.array([0.5])
            self.info = {"sfreq": 125.0}
            self.ch_names = ["E1"]

        def crop(self, tmin, tmax):
            return self

        def filter(self, l_freq, h_freq, verbose=False):
            return self

        def resample(self, sfreq, verbose=False):
            self.info["sfreq"] = sfreq
            return self

        def get_data(self):
            # 少于一个 10 秒窗口，确保 num_windows=0
            return np.zeros((1, 100))

    fake_mne = SimpleNamespace(
        io=SimpleNamespace(read_raw_edf=lambda *args, **kwargs: FakeRaw())
    )
    monkeypatch.setattr(mod, "mne", fake_mne)

    with caplog.at_level(logging.INFO, logger="run_modma_real"):
        with pytest.raises(ValueError, match="未能成功加载任何有效数据"):
            mod.load_real_modma_data(max_subjects=2, crop_duration=60)

    assert "已加载 0 名受试者" not in caplog.text


def test_load_real_data_success_path_structure(monkeypatch):
    """验证成功路径：labels/groups 对齐，EpochsArray 维度与通道一致。"""
    df = pd.DataFrame({
        "participant_id": ["sub-001", "sub-025"],
        "group": ["MDD", "HC"],
    })

    monkeypatch.setattr(mod.os.path, "exists", lambda _: True)
    monkeypatch.setattr(mod.pd, "read_csv", lambda *args, **kwargs: df.copy())
    monkeypatch.setattr(mod.glob, "glob", lambda pattern: ["fake.edf"])

    N_CH, N_SAMP = 2, 7500  # 60 sec @ 125 Hz → 6 windows per subject

    class FakeRaw:
        def __init__(self):
            self.times = np.linspace(0, 59.99, N_SAMP)
            self.info = {"sfreq": 125.0}
            self.ch_names = ["E1", "E2"]

        def crop(self, tmin, tmax): return self
        def filter(self, l_freq, h_freq, verbose=False): return self
        def resample(self, sfreq, verbose=False): return self
        def get_data(self): return np.zeros((N_CH, N_SAMP))

    class FakeEpochsArray:
        def __init__(self, data, info, verbose=False):
            self._data = data
        def get_data(self): return self._data

    fake_mne = SimpleNamespace(
        io=SimpleNamespace(read_raw_edf=lambda *args, **kwargs: FakeRaw()),
        create_info=lambda ch_names, sfreq, ch_types: SimpleNamespace(),
        EpochsArray=FakeEpochsArray,
    )
    monkeypatch.setattr(mod, "mne", fake_mne)

    epochs, labels, groups = mod.load_real_modma_data(max_subjects=2, crop_duration=60)

    assert len(labels) == len(groups), "labels 与 groups 长度不一致"
    assert epochs.get_data().shape[0] == len(labels), "EpochsArray 试次数与 labels 不一致"
    assert epochs.get_data().shape[1] == N_CH, "EpochsArray 通道数与原始数据不一致"
    assert set(labels.tolist()) == {0, 1}, "成功路径应同时包含 MDD(1) 和 HC(0) 标签"
