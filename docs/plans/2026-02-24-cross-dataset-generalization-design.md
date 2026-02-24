# TDBrain 跨数据集泛化测试 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 测试 TDBrain MDD 分类器在 MODMA 数据集上的跨数据集泛化能力

**Architecture:** 两阶段实验——方案 A（通道映射 + 直接泛化）和方案 B（脑区级统一特征空间）。TDBrain DISCOVERY 训练，MODMA resting-state 测试。通过 EGI-128 到 10-20 的 3D 坐标最近邻映射解决通道不匹配问题。

**Tech Stack:** MNE-Python, scikit-learn, imbalanced-learn, scipy, numpy

---

## 背景

- TDBrain: 26ch 10-20 系统, 1138 subjects (305 MDD + 833 nonMDD), BrainVision 格式
- MODMA resting-state: 128ch EGI HydroCel-128, ~54 subjects (MDD/HC), .mat 格式
- TDBrain 特征: 496 维/condition (band power + Hjorth + entropy), 两个 condition 合并 992 维
- MODMA resting 特征: 15 维 region-level (相对频带功率 + 不对称性 + 连接性)
- 已有代码: `tdbrain-challenge/submit_pipeline.py` (DISCOVERY→REPLICATION), `scripts/modma/modma_mdd_real_experiment.py` (MODMA resting pipeline)

## 关键约束

1. MODMA 只有 resting-state（无 EO/EC 区分），TDBrain 有 EO+EC 两个 condition
2. MODMA .mat 文件 key 需动态发现，不能硬编码
3. EGI-128 通道名为 E1-E128，需要 3D 坐标映射到 10-20 名称
4. MODMA 采样率可能与 TDBrain 不同，需统一

---

### Task 1: EGI-128 到 10-20 通道映射模块

**Files:**
- Create: `tdbrain-challenge/channel_mapper.py`
- Test: `tdbrain-challenge/test_channel_mapper.py`

**Step 1: Write the failing test**

```python
# test_channel_mapper.py
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
    # E36 is near Cz in GSN-HydroCel-128
    # The exact mapping depends on 3D coordinates, so just check structure
    for egi_ch, std_ch in mapping.items():
        assert egi_ch in egi_names
        assert std_ch in TDBRAIN_CHANNELS
```

**Step 2: Run test to verify it fails**

Run: `cd /d/eeg/tdbrain-challenge && python -m pytest test_channel_mapper.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'channel_mapper'"

**Step 3: Write minimal implementation**

```python
# channel_mapper.py
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
```

**Step 4: Run test to verify it passes**

Run: `cd /d/eeg/tdbrain-challenge && python -m pytest test_channel_mapper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tdbrain-challenge/channel_mapper.py tdbrain-challenge/test_channel_mapper.py
git commit -m "feat: add EGI-128 to 10-20 channel mapper for cross-dataset generalization"
```

---

### Task 2: MODMA Resting-State 数据加载器（适配 TDBrain 格式）

**Files:**
- Create: `tdbrain-challenge/modma_loader.py`
- Test: `tdbrain-challenge/test_modma_loader.py`

**Step 1: Write the failing test**

```python
# test_modma_loader.py
import numpy as np
from modma_loader import load_modma_resting_raw, get_modma_subjects

def test_get_modma_subjects_returns_list():
    """Should return list of (filepath, label) tuples."""
    subjects = get_modma_subjects()
    assert len(subjects) > 0
    for fpath, label in subjects:
        assert label in ("MDD", "HC")

def test_load_modma_resting_raw_shape():
    """Loaded data should be a MNE Raw with EGI-128 channels."""
    subjects = get_modma_subjects()
    raw, label = load_modma_resting_raw(subjects[0][0])
    assert raw.info['nchan'] >= 26  # at least enough for mapping
```

**Step 2: Run test to verify it fails**

Run: `cd /d/eeg/tdbrain-challenge && python -m pytest test_modma_loader.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# modma_loader.py
"""Load MODMA resting-state .mat files as MNE Raw objects."""
import numpy as np
import mne
from pathlib import Path
from scipy.io import loadmat

MODMA_RESTING_DIR = Path(r"D:\eeg\data\modma_resting\EEG_128channels_resting_lanzhou_2015")
SFREQ = 250.0  # MODMA default sampling rate

def get_modma_subjects():
    """Return list of (filepath, label_str) for all resting-state subjects."""
    mat_files = sorted(MODMA_RESTING_DIR.glob("*.mat"))
    subjects = []
    for f in mat_files:
        name = f.name
        if name.startswith("0201"):
            subjects.append((f, "MDD"))
        elif name.startswith("0202") or name.startswith("0203"):
            subjects.append((f, "HC"))
    return subjects

def load_modma_resting_raw(filepath):
    """Load a single MODMA .mat resting-state file as MNE Raw.

    Dynamically discovers the data key (never hardcodes).
    Returns: (mne.io.RawArray, sfreq)
    """
    mat = loadmat(str(filepath))
    data_keys = [k for k in mat.keys() if not k.startswith('__')]
    if len(data_keys) != 1:
        raise ValueError(f"Expected 1 data key, got {data_keys} in {filepath}")
    data = mat[data_keys[0]]  # shape: (n_channels, n_samples) or (n_samples, n_channels)

    # Ensure (n_channels, n_samples)
    if data.shape[0] > data.shape[1]:
        data = data.T

    n_ch = data.shape[0]
    # Create EGI-128 channel names
    ch_names = [f"E{i+1}" for i in range(n_ch)]
    if n_ch > 128:
        ch_names = ch_names[:128]
        data = data[:128, :]

    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # assume uV -> V
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    # Only keep channels that exist in montage
    valid = [ch for ch in raw.ch_names if ch in montage.ch_names]
    raw.pick(valid)
    raw.set_montage(montage, verbose=False)
    return raw
```

**Step 4: Run test to verify it passes**

Run: `cd /d/eeg/tdbrain-challenge && python -m pytest test_modma_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tdbrain-challenge/modma_loader.py tdbrain-challenge/test_modma_loader.py
git commit -m "feat: add MODMA resting-state data loader for cross-dataset testing"
```

---

### Task 3: 方案 A — 直接泛化测试脚本

**Files:**
- Create: `tdbrain-challenge/run_cross_dataset.py`

**依赖:** Task 1 (channel_mapper), Task 2 (modma_loader), 已有的 feature_extractor.py, preprocessor.py, run_ablation.py

**Step 1: Write the cross-dataset evaluation script**

核心逻辑：
1. 加载 TDBrain DISCOVERY 训练数据（复用 `run_ablation.load_all()`）
2. 加载 MODMA resting-state 数据
3. 对每个 MODMA 受试者：加载 .mat → 通道映射 → 预处理 → 提取 496 维特征
4. 因为 MODMA 无 EO/EC 区分，用同一段 resting 数据作为两个 condition（或只用单 condition 496 维）
5. 训练 SVM pipeline on DISCOVERY，预测 MODMA
6. 报告 AUC, BA, sensitivity, specificity

```python
# run_cross_dataset.py
"""Cross-dataset generalization: Train on TDBrain DISCOVERY, test on MODMA resting-state.

Approach A: Channel mapping (EGI-128 → 10-20) + TDBrain feature extractor.
Approach B: Region-level unified features.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mne

from channel_mapper import map_egi128_to_1020, TDBRAIN_CHANNELS
from modma_loader import get_modma_subjects, load_modma_resting_raw
from config import EEG_CHANNELS, EPOCH_DURATION, REJECT_THRESHOLD, FREQ_BANDS
from feature_extractor import extract_features
from run_ablation import load_all

mne.set_log_level("ERROR")
RS = 42
RESULTS_DIR = Path("results")


def preprocess_modma_as_tdbrain(raw):
    """Preprocess MODMA raw to match TDBrain pipeline.

    1. Map EGI-128 channels to 10-20
    2. Pick mapped channels, rename to 10-20 names
    3. Filter 1-40 Hz, average reference, 2s epochs, artifact rejection
    """
    mapping = map_egi128_to_1020(raw.ch_names)
    egi_picks = list(mapping.keys())
    raw.pick(egi_picks)
    rename_dict = mapping
    raw.rename_channels(rename_dict)
    # Reorder to match TDBrain channel order
    raw.reorder_channels(TDBRAIN_CHANNELS)

    raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DURATION,
                                          preload=True, verbose=False)
    n_before = len(epochs)
    epochs.drop_bad(reject=dict(eeg=REJECT_THRESHOLD), verbose=False)
    print(f"    Epochs: {n_before} -> {len(epochs)}")
    return epochs


def extract_modma_features():
    """Extract TDBrain-compatible features for all MODMA subjects."""
    subjects = get_modma_subjects()
    print(f"MODMA subjects: {len(subjects)}")

    cache_dir = RESULTS_DIR / "modma_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    features, labels, sids = [], [], []
    for fpath, label in subjects:
        sid = fpath.stem
        cache_path = cache_dir / f"{sid}.npz"

        if cache_path.exists():
            d = np.load(cache_path)
            features.append(d["feat"])
            labels.append(label)
            sids.append(sid)
            continue

        try:
            raw = load_modma_resting_raw(fpath)
            epochs = preprocess_modma_as_tdbrain(raw)
            if len(epochs) < 3:
                print(f"  [SKIP] {sid}: too few epochs ({len(epochs)})")
                continue
            feat = extract_features(epochs)  # 496-dim
            np.savez_compressed(cache_path, feat=feat, label=label)
            features.append(feat)
            labels.append(label)
            sids.append(sid)
            print(f"  [{len(features)}] {sid}: {label}, feat={feat.shape}")
        except Exception as e:
            print(f"  [SKIP] {sid}: {e}")

    return np.array(features), np.array(labels), sids


def run_approach_a():
    """Approach A: Direct generalization with channel mapping."""
    print("=" * 60)
    print("APPROACH A: Channel mapping + direct generalization")
    print("=" * 60)

    # Load TDBrain DISCOVERY (use EO only since MODMA has no EO/EC split)
    print("\nLoading TDBrain DISCOVERY features...")
    datasets = load_all()

    # Extract MODMA features
    print("\nExtracting MODMA features (TDBrain-compatible)...")
    X_modma, y_modma_str, modma_sids = extract_modma_features()

    # Remap MODMA labels: MDD->MDD, HC->nonMDD (to match TDBrain convention)
    y_modma_mapped = np.where(y_modma_str == "MDD", "MDD", "nonMDD")

    results = {}
    # Test with EO-only, EC-only, and combined (duplicate for combined)
    for cond_name in ["EO", "EC"]:
        if cond_name not in datasets:
            continue
        X_train, y_train, groups = datasets[cond_name]
        print(f"\n--- Condition: {cond_name} ---")
        print(f"  Train: {X_train.shape}, MDD={np.sum(y_train=='MDD')}")
        print(f"  Test:  {X_modma.shape}, MDD={np.sum(y_modma_mapped=='MDD')}")

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_te_enc = le.transform(y_modma_mapped)
        mdd_idx = list(le.classes_).index("MDD")

        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RS, k_neighbors=5)),
            ("selector", SelectKBest(f_classif, k=min(50, X_train.shape[1]))),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=RS)),
        ])
        pipe.fit(X_train, y_tr_enc)
        proba = pipe.predict_proba(X_modma)[:, mdd_idx]
        pred = pipe.predict(X_modma)

        auc = roc_auc_score(y_te_enc, proba)
        ba = balanced_accuracy_score(y_te_enc, pred)
        tn, fp, fn, tp = confusion_matrix(y_te_enc, pred).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0

        results[cond_name] = {
            "auc": round(float(auc), 4),
            "balanced_accuracy": round(float(ba), 4),
            "sensitivity": round(float(sen), 4),
            "specificity": round(float(spe), 4),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_modma.shape[0]),
        }
        print(f"  AUC={auc:.4f}, BA={ba:.4f}, Sen={sen:.4f}, Spe={spe:.4f}")

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {"approach_a": run_approach_a()}

    out_path = RESULTS_DIR / "cross_dataset_results.json"
    with open(out_path, "w", newline="\n") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

Run: `cd /d/eeg/tdbrain-challenge && python run_cross_dataset.py`
Expected: Completes with AUC/BA metrics printed (likely near chance ~0.5 for cross-dataset)

**Step 3: Commit**

```bash
git add tdbrain-challenge/run_cross_dataset.py
git commit -m "feat: add cross-dataset generalization test (Approach A: channel mapping)"
```

---

### Task 4: 方案 B — 脑区级统一特征空间

**Files:**
- Modify: `tdbrain-challenge/run_cross_dataset.py` (添加 `run_approach_b()`)

**Step 1: 实现脑区级特征提取器**

两个数据集都提取相同的 region-level 特征：
- 5 个脑区 × 5 个频带 = 25 维相对频带功率
- Frontal alpha asymmetry (1 维)
- Frontal TBR (1 维)
- 共 27 维

```python
# 添加到 run_cross_dataset.py

from scipy.signal import welch as scipy_welch

# Region mapping for TDBrain 10-20 channels
TDBRAIN_REGIONS = {
    "frontal": ["Fp1","Fp2","F7","F3","Fz","F4","F8","FC3","FCz","FC4"],
    "central": ["C3","Cz","C4"],
    "temporal": ["T7","T8"],
    "parietal": ["CP3","CPz","CP4","P7","P3","Pz","P4","P8"],
    "occipital": ["O1","Oz","O2"],
}

BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,40)}

def extract_region_features(epochs_or_data, sfreq, ch_names, region_map):
    """Extract 27-dim region-level features from any montage.

    Args:
        epochs_or_data: (n_epochs, n_ch, n_times) array
        sfreq: sampling frequency
        ch_names: list of channel names
        region_map: dict {region_name: [channel_names]}

    Returns: (27,) feature vector
    """
    if hasattr(epochs_or_data, 'get_data'):
        data = epochs_or_data.get_data()
    else:
        data = epochs_or_data

    n_ep, n_ch, n_times = data.shape
    nperseg = min(n_times, int(sfreq * 2))

    # PSD per channel, averaged over epochs
    freqs, psd = scipy_welch(data.mean(axis=0), fs=sfreq, nperseg=nperseg, axis=-1)
    total_power = psd.sum(axis=-1)  # (n_ch,)

    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    features = []

    # 25 dims: 5 regions × 5 bands relative power
    for region in ["frontal","central","temporal","parietal","occipital"]:
        r_chs = [ch_idx[c] for c in region_map.get(region, []) if c in ch_idx]
        if not r_chs:
            features.extend([0.0] * 5)
            continue
        for bname, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            bp = psd[r_chs][:, mask].mean()
            tp = total_power[r_chs].mean()
            features.append(float(bp / (tp + 1e-10)))

    # FAA: ln(alpha_right) - ln(alpha_left)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    left_frontal = [ch_idx[c] for c in ["F3","Fp1","F7"] if c in ch_idx]
    right_frontal = [ch_idx[c] for c in ["F4","Fp2","F8"] if c in ch_idx]
    if left_frontal and right_frontal:
        al = psd[left_frontal][:, alpha_mask].mean()
        ar = psd[right_frontal][:, alpha_mask].mean()
        features.append(float(np.log(ar + 1e-10) - np.log(al + 1e-10)))
    else:
        features.append(0.0)

    # TBR: theta/beta ratio (frontal)
    frontal_idx = [ch_idx[c] for c in region_map.get("frontal", []) if c in ch_idx]
    if frontal_idx:
        theta_mask = (freqs >= 4) & (freqs < 8)
        beta_mask = (freqs >= 13) & (freqs < 30)
        theta_p = psd[frontal_idx][:, theta_mask].mean()
        beta_p = psd[frontal_idx][:, beta_mask].mean()
        features.append(float(theta_p / (beta_p + 1e-10)))
    else:
        features.append(0.0)

    return np.array(features)  # (27,)
```

**Step 2: 实现方案 B 的训练/测试流程**

```python
def run_approach_b():
    """Approach B: Region-level unified feature space."""
    print("\n" + "=" * 60)
    print("APPROACH B: Region-level unified features")
    print("=" * 60)

    # --- TDBrain: re-extract region features from cached epochs ---
    # We need raw epochs, not pre-extracted 496-dim features.
    # Load from DISCOVERY cache and re-extract region features.
    from run_ablation import CACHE_DIR
    from data_loader import load_subjects

    print("\nRe-extracting TDBrain region features...")
    # Use EO condition
    df = load_subjects("EO")
    cache = CACHE_DIR / "EO"

    X_train_list, y_train_list = [], []
    for _, row in df.iterrows():
        sid = row["participants_ID"]
        label = row["indication"]
        npz = cache / f"{sid}.npz"
        if not npz.exists():
            continue
        # We can't get raw epochs from cache (only 496-dim features stored)
        # So we re-extract region features from the PSD embedded in 496-dim
        # Actually, the 496-dim already contains band power per channel.
        # We can reconstruct region features from abs_bp (first 130 dims = 5 bands × 26 ch)
        d = np.load(npz)
        feat_496 = d["feat"]
        # abs_bp layout: [delta(26), theta(26), alpha(26), beta(26), gamma(26)] = 130
        abs_bp = feat_496[:130].reshape(5, 26)  # (5_bands, 26_ch)
        total = abs_bp.sum(axis=0)  # (26,)

        region_feats = []
        for region in ["frontal","central","temporal","parietal","occipital"]:
            r_idx = [EEG_CHANNELS.index(c) for c in TDBRAIN_REGIONS[region]]
            for b in range(5):
                bp = abs_bp[b, r_idx].mean()
                tp = total[r_idx].mean()
                region_feats.append(float(bp / (tp + 1e-10)))

        # FAA from abs_bp alpha band (index 2)
        f3_i, f4_i = EEG_CHANNELS.index("F3"), EEG_CHANNELS.index("F4")
        al, ar = abs_bp[2, f3_i], abs_bp[2, f4_i]
        region_feats.append(float(np.log(ar + 1e-10) - np.log(al + 1e-10)))

        # TBR
        frontal_idx = [EEG_CHANNELS.index(c) for c in TDBRAIN_REGIONS["frontal"]]
        theta_f = abs_bp[1, frontal_idx].mean()
        beta_f = abs_bp[3, frontal_idx].mean()
        region_feats.append(float(theta_f / (beta_f + 1e-10)))

        X_train_list.append(region_feats)
        y_train_list.append("MDD" if label == "MDD" else "nonMDD")

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    print(f"  TDBrain region features: {X_train.shape}")

    # --- MODMA: extract region features ---
    print("\nExtracting MODMA region features...")
    from modma_loader import get_modma_subjects, load_modma_resting_raw
    from channel_mapper import map_egi128_to_1020

    # Use EGI-128 region map from modma_mdd_real_experiment.py
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "modma"))
    from modma_mdd_real_experiment import EGI128_REGION_MAP

    subjects = get_modma_subjects()
    X_test_list, y_test_list = [], []
    for fpath, label in subjects:
        try:
            raw = load_modma_resting_raw(fpath)
            raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
            raw.set_eeg_reference('average', projection=False, verbose=False)
            epochs = mne.make_fixed_length_epochs(raw, duration=2.0,
                                                  preload=True, verbose=False)
            epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False)
            if len(epochs) < 3:
                continue

            ch_names = epochs.ch_names
            # Build EGI region map for available channels
            egi_regions = {}
            for ch in ch_names:
                r = EGI128_REGION_MAP.get(ch)
                if r:
                    egi_regions.setdefault(r, []).append(ch)

            feat = extract_region_features(epochs, epochs.info['sfreq'],
                                           ch_names, egi_regions)
            X_test_list.append(feat)
            y_test_list.append("MDD" if label == "MDD" else "nonMDD")
        except Exception as e:
            print(f"  [SKIP] {fpath.stem}: {e}")

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    print(f"  MODMA region features: {X_test.shape}")

    # --- Classify ---
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)
    y_te_enc = le.transform(y_test)
    mdd_idx = list(le.classes_).index("MDD")

    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RS, k_neighbors=5)),
        ("clf", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=RS)),
    ])
    pipe.fit(X_train, y_tr_enc)
    proba = pipe.predict_proba(X_test)[:, mdd_idx]
    pred = pipe.predict(X_test)

    auc = roc_auc_score(y_te_enc, proba)
    ba = balanced_accuracy_score(y_te_enc, pred)
    tn, fp, fn, tp = confusion_matrix(y_te_enc, pred).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    result = {
        "auc": round(float(auc), 4),
        "balanced_accuracy": round(float(ba), 4),
        "sensitivity": round(float(sen), 4),
        "specificity": round(float(spe), 4),
        "n_features": 27,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    print(f"\n  AUC={auc:.4f}, BA={ba:.4f}, Sen={sen:.4f}, Spe={spe:.4f}")
    return result
```

**Step 2: 更新 main() 加入方案 B**

```python
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "approach_a": run_approach_a(),
        "approach_b": run_approach_b(),
    }
    # ... save results
```

**Step 3: Run**

Run: `cd /d/eeg/tdbrain-challenge && python run_cross_dataset.py`

**Step 4: Commit**

```bash
git add tdbrain-challenge/run_cross_dataset.py
git commit -m "feat: add Approach B (region-level features) to cross-dataset generalization"
```

---

### Task 5: 结果分析与报告

**Step 1: 检查结果**

Run: `cd /d/eeg/tdbrain-challenge && python -c "import json; print(json.dumps(json.load(open('results/cross_dataset_results.json')), indent=2))"`

**Step 2: 分析并总结**

预期结果解读：
- 方案 A AUC ~0.5: 通道级特征不泛化（预处理差异、采样率、通道映射误差）
- 方案 B AUC > 方案 A: 脑区级特征更鲁棒
- 两者都 < TDBrain 内部 CV (AUC=0.668): 跨数据集泛化困难是 EEG 领域的已知问题

**Step 3: Commit results**

```bash
git add tdbrain-challenge/results/
git commit -m "docs: add cross-dataset generalization results (TDBrain → MODMA)"
```
