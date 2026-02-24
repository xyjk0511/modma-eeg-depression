"""Cross-dataset generalization: Train on TDBrain DISCOVERY, test on MODMA resting-state.

Approach A: Channel mapping (EGI-128 -> 10-20) + TDBrain feature extractor (496-dim).
Approach B: Region-level unified features (27-dim).
"""
import json
import numpy as np
from pathlib import Path
from scipy.signal import welch as scipy_welch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mne

from channel_mapper import map_egi128_to_1020, TDBRAIN_CHANNELS
from modma_loader import get_modma_subjects, load_modma_resting_raw
from config import EEG_CHANNELS, EPOCH_DURATION, REJECT_THRESHOLD
from feature_extractor import extract_features
from run_ablation import load_all, CACHE_DIR
from main_diagnostic import load_discovery_subjects

mne.set_log_level("ERROR")
RS = 42
RESULTS_DIR = Path("results")

# ── Region mappings ──────────────────────────────────────────────
TDBRAIN_REGIONS = {
    "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4"],
    "central": ["C3", "Cz", "C4"],
    "temporal": ["T7", "T8"],
    "parietal": ["CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8"],
    "occipital": ["O1", "Oz", "O2"],
}

REGION_ORDER = ["frontal", "central", "temporal", "parietal", "occipital"]
BAND_DEFS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
             "beta": (13, 30), "gamma": (30, 40)}

# EGI-128 region map (from modma_mdd_real_experiment.py)
def _egi_range(start, end):
    return [f"E{i}" for i in range(start, end + 1)]

EGI128_REGION_MAP = {}
for ch in _egi_range(1, 4) + _egi_range(8, 14) + _egi_range(19, 24) + _egi_range(26, 28) + ["E33", "E34"] + _egi_range(117, 128):
    EGI128_REGION_MAP[ch] = "frontal"
for ch in _egi_range(5, 7) + _egi_range(29, 32) + _egi_range(35, 37) + ["E54", "E55", "E79", "E80"] + _egi_range(104, 106):
    EGI128_REGION_MAP[ch] = "central"
for ch in _egi_range(38, 46) + _egi_range(96, 103) + _egi_range(107, 116):
    EGI128_REGION_MAP[ch] = "temporal"
for ch in _egi_range(47, 53) + _egi_range(56, 62) + ["E77", "E78"] + _egi_range(85, 95):
    EGI128_REGION_MAP[ch] = "parietal"
for ch in _egi_range(63, 76) + _egi_range(81, 84):
    EGI128_REGION_MAP[ch] = "occipital"


# ── Approach A: Channel-level direct generalization ──────────────
def preprocess_modma_as_tdbrain(raw):
    """Preprocess MODMA raw to match TDBrain pipeline exactly."""
    mapping = map_egi128_to_1020(raw.ch_names)
    egi_picks = list(mapping.keys())
    raw.pick(egi_picks)
    raw.rename_channels(mapping)
    raw.reorder_channels(TDBRAIN_CHANNELS)
    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_DURATION, preload=True, verbose=False
    )
    n_before = len(epochs)
    epochs.drop_bad(reject=dict(eeg=REJECT_THRESHOLD), verbose=False)
    print(f"    Epochs: {n_before} -> {len(epochs)}")
    return epochs


def extract_modma_features():
    """Extract TDBrain-compatible 496-dim features for all MODMA subjects."""
    subjects = get_modma_subjects()
    print(f"MODMA subjects found: {len(subjects)}")

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
            feat = extract_features(epochs)
            np.savez_compressed(cache_path, feat=feat, label=label)
            features.append(feat)
            labels.append(label)
            sids.append(sid)
            print(f"  [{len(features)}] {sid}: {label}, feat={feat.shape}")
        except Exception as e:
            print(f"  [SKIP] {sid}: {e}")

    return np.array(features), np.array(labels), sids


def _train_and_eval(X_train, y_train_str, X_test, y_test_str, k=50):
    """Train SVM on TDBrain, evaluate on MODMA. Returns metrics dict."""
    # Explicit binary encoding: MDD=1, nonMDD/HC=0
    y_tr = np.where(np.array(y_train_str) == "MDD", 1, 0)
    y_te = np.where(np.array(y_test_str) == "MDD", 1, 0)

    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RS, k_neighbors=5)),
        ("selector", SelectKBest(f_classif, k=min(k, X_train.shape[1]))),
        ("clf", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=RS)),
    ])
    pipe.fit(X_train, y_tr)
    proba = pipe.predict_proba(X_test)[:, 1]  # P(MDD)
    pred = pipe.predict(X_test)

    auc = roc_auc_score(y_te, proba)
    ba = balanced_accuracy_score(y_te, pred)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "auc": round(float(auc), 4),
        "balanced_accuracy": round(float(ba), 4),
        "sensitivity": round(float(sen), 4),
        "specificity": round(float(spe), 4),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_in": int(X_train.shape[1]),
    }


def run_approach_a():
    """Approach A: Channel mapping + TDBrain 496-dim features."""
    print("=" * 60)
    print("APPROACH A: Channel mapping + direct generalization")
    print("=" * 60)

    print("\nLoading TDBrain DISCOVERY features...")
    datasets = load_all()

    print("\nExtracting MODMA features (TDBrain-compatible 496-dim)...")
    X_modma, y_modma, modma_sids = extract_modma_features()
    print(f"MODMA: {X_modma.shape}, MDD={np.sum(y_modma=='MDD')}, HC={np.sum(y_modma=='HC')}")

    results = {}
    for cond in ["EO", "EC"]:
        if cond not in datasets:
            continue
        X_train, y_train, _ = datasets[cond]
        print(f"\n--- {cond}: Train {X_train.shape}, Test {X_modma.shape} ---")
        r = _train_and_eval(X_train, y_train, X_modma, y_modma, k=50)
        results[cond] = r
        print(f"  AUC={r['auc']}, BA={r['balanced_accuracy']}, "
              f"Sen={r['sensitivity']}, Spe={r['specificity']}")

    return results


# ── Approach B: Region-level unified features ────────────────────
def extract_region_features_from_psd(psd, freqs, ch_names, region_map):
    """Extract 27-dim region-level features from PSD array.

    25 dims: 5 regions x 5 bands (relative power)
    1 dim: frontal alpha asymmetry
    1 dim: frontal theta/beta ratio
    """
    total_power = psd.sum(axis=-1)
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    features = []

    for region in REGION_ORDER:
        r_chs = [ch_idx[c] for c in region_map.get(region, []) if c in ch_idx]
        if not r_chs:
            features.extend([0.0] * 5)
            continue
        for _, (fmin, fmax) in BAND_DEFS.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            bp = psd[r_chs][:, mask].mean()
            tp = total_power[r_chs].mean()
            features.append(float(bp / (tp + 1e-10)))

    # FAA
    alpha_mask = (freqs >= 8) & (freqs < 13)
    left_f = [ch_idx[c] for c in ["F3", "Fp1", "F7"] if c in ch_idx]
    right_f = [ch_idx[c] for c in ["F4", "Fp2", "F8"] if c in ch_idx]
    if left_f and right_f:
        al = psd[left_f][:, alpha_mask].mean()
        ar = psd[right_f][:, alpha_mask].mean()
        features.append(float(np.log(ar + 1e-10) - np.log(al + 1e-10)))
    else:
        features.append(0.0)

    # TBR
    frontal_chs = [ch_idx[c] for c in region_map.get("frontal", []) if c in ch_idx]
    if frontal_chs:
        theta_mask = (freqs >= 4) & (freqs < 8)
        beta_mask = (freqs >= 13) & (freqs < 30)
        theta_p = psd[frontal_chs][:, theta_mask].mean()
        beta_p = psd[frontal_chs][:, beta_mask].mean()
        features.append(float(theta_p / (beta_p + 1e-10)))
    else:
        features.append(0.0)

    return np.array(features)


def _tdbrain_region_from_cache(condition):
    """Extract 27-dim region features from TDBrain cached 496-dim features."""
    df = load_discovery_subjects(condition)
    cache = CACHE_DIR / condition

    X_list, y_list = [], []
    for _, row in df.iterrows():
        sid = row["participants_ID"]
        label = row["indication"]
        npz = cache / f"{sid}.npz"
        if not npz.exists():
            continue
        feat_496 = np.load(npz)["feat"]
        # abs_bp: first 130 dims = 5 bands x 26 channels
        abs_bp = feat_496[:130].reshape(5, 26)
        total = abs_bp.sum(axis=0)

        region_feats = []
        for region in REGION_ORDER:
            r_idx = [EEG_CHANNELS.index(c) for c in TDBRAIN_REGIONS[region]]
            for b in range(5):
                bp = abs_bp[b, r_idx].mean()
                tp = total[r_idx].mean()
                region_feats.append(float(bp / (tp + 1e-10)))

        # FAA from alpha band (index 2)
        f3_i, f4_i = EEG_CHANNELS.index("F3"), EEG_CHANNELS.index("F4")
        region_feats.append(float(
            np.log(abs_bp[2, f4_i] + 1e-10) - np.log(abs_bp[2, f3_i] + 1e-10)
        ))

        # TBR
        frontal_idx = [EEG_CHANNELS.index(c) for c in TDBRAIN_REGIONS["frontal"]]
        region_feats.append(float(
            abs_bp[1, frontal_idx].mean() / (abs_bp[3, frontal_idx].mean() + 1e-10)
        ))

        X_list.append(region_feats)
        y_list.append("MDD" if label == "MDD" else "nonMDD")

    return np.array(X_list), np.array(y_list)


def _modma_region_features():
    """Extract 27-dim region features from MODMA resting-state data."""
    subjects = get_modma_subjects()
    X_list, y_list = [], []

    for fpath, label in subjects:
        sid = fpath.stem
        try:
            raw = load_modma_resting_raw(fpath)
            raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)
            raw.set_eeg_reference("average", projection=False, verbose=False)
            epochs = mne.make_fixed_length_epochs(
                raw, duration=2.0, preload=True, verbose=False
            )
            epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False)
            if len(epochs) < 3:
                print(f"  [SKIP] {sid}: too few epochs")
                continue

            data = epochs.get_data()
            sfreq = epochs.info["sfreq"]
            ch_names = epochs.ch_names
            nperseg = min(data.shape[2], int(sfreq * 2))
            freqs, psd = scipy_welch(data.mean(axis=0), fs=sfreq,
                                     nperseg=nperseg, axis=-1)

            # Build EGI region map for available channels
            egi_regions = {}
            for ch in ch_names:
                r = EGI128_REGION_MAP.get(ch)
                if r:
                    egi_regions.setdefault(r, []).append(ch)

            # For FAA/TBR we need standard names; map EGI to 10-20 for those
            mapping_1020 = map_egi128_to_1020(ch_names)
            ch_names_mapped = list(ch_names)
            for egi_ch, std_ch in mapping_1020.items():
                if egi_ch in ch_names:
                    idx = ch_names_mapped.index(egi_ch)
                    ch_names_mapped[idx] = std_ch

            feat = extract_region_features_from_psd(
                psd, freqs, ch_names_mapped, egi_regions
            )
            X_list.append(feat)
            y_list.append("MDD" if label == "MDD" else "nonMDD")
        except Exception as e:
            print(f"  [SKIP] {sid}: {e}")

    return np.array(X_list), np.array(y_list)


def run_approach_b():
    """Approach B: Region-level unified feature space (27-dim)."""
    print("\n" + "=" * 60)
    print("APPROACH B: Region-level unified features (27-dim)")
    print("=" * 60)

    results = {}
    for cond in ["EO", "EC"]:
        print(f"\n--- {cond} ---")
        print("  Extracting TDBrain region features from cache...")
        X_train, y_train = _tdbrain_region_from_cache(cond)
        print(f"  TDBrain: {X_train.shape}, MDD={np.sum(y_train=='MDD')}")

        print("  Extracting MODMA region features...")
        X_test, y_test = _modma_region_features()
        print(f"  MODMA: {X_test.shape}, MDD={np.sum(y_test=='MDD')}")

        r = _train_and_eval(X_train, y_train, X_test, y_test, k=min(20, 27))
        results[cond] = r
        print(f"  AUC={r['auc']}, BA={r['balanced_accuracy']}, "
              f"Sen={r['sensitivity']}, Spe={r['specificity']}")

    return results


# ── Main ─────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {
        "approach_a": run_approach_a(),
        "approach_b": run_approach_b(),
    }

    out_path = RESULTS_DIR / "cross_dataset_results.json"
    with open(out_path, "w", newline="\n") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for approach, data in results.items():
        print(f"\n{approach}:")
        for cond, metrics in data.items():
            print(f"  {cond}: AUC={metrics['auc']}, BA={metrics['balanced_accuracy']}")


if __name__ == "__main__":
    main()
