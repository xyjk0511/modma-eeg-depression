"""MODMA ERP (Dot-probe hcue) subject-level classification.

P300 features: mean amplitude in 250-500ms window per channel, computed from
the average ERP (mean across trials per subject). One vector per subject.
Reference: Li et al. 2018, CMPB — ERP waveform shape features for MDD classification.

Usage: python run_modma_erp.py
"""
import os, gc, re, csv, warnings, numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import ttest_ind, false_discovery_control

import mne
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ── Config ──────────────────────────────────────────────────────
ERP_DIR = Path("854301_EEG_128Channels_ERP_Lanzhou_2015/EEG_128channels_ERP_lanzhou_2015")
CONDITION = "hcue"          # overridden in __main__ loop
TMIN, TMAX = -0.1, 0.5     # epoch window (sec)
BASELINE = (-0.1, 0.0)     # baseline correction
FMIN, FMAX = 0.5, 40.0     # bandpass filter
N_EEG_CH = 128
P300_WIN = (0.25, 0.50)    # P300 time window (sec)
N200_WIN = (0.10, 0.25)    # N200 time window (sec)
RESTING_DIR = Path("854301_EEG_128Channels_Resting_Lanzhou_2015/EEG_128channels_resting_lanzhou_2015")
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}

BINS = [
    ("250-300ms", 0.250, 0.300),
    ("300-350ms", 0.300, 0.350),
    ("350-400ms", 0.350, 0.400),
    ("400-450ms", 0.400, 0.450),
    ("450-500ms", 0.450, 0.500),
]


def get_parietal_indices(targets=("Pz", "P3", "P4", "Cz", "CPz")):
    """Return {name: channel_index} for EGI HydroCel-128 montage."""
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    name_to_idx = {n: i for i, n in enumerate(montage.ch_names)}
    result = {}
    for t in targets:
        if t in name_to_idx:
            result[t] = name_to_idx[t]
            print(f"  {t}: index={result[t]}", flush=True)
        else:
            ref = mne.channels.make_standard_montage("standard_1020")
            ref_pos = dict(zip(ref.ch_names, ref.get_positions()["ch_pos"].values()))
            gsn_pos = montage.get_positions()["ch_pos"]
            if t not in ref_pos:
                raise ValueError(f"{t} not in standard_1020")
            tp = ref_pos[t]
            nearest = min(gsn_pos, key=lambda n: np.linalg.norm(gsn_pos[n] - tp))
            result[t] = name_to_idx[nearest]
            print(f"  {t} -> nearest GSN: {nearest} (idx={result[t]})", flush=True)
    return result


def load_erp_features():
    """One feature vector per subject: mean P300 amplitude per channel."""
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    print(f"Found {len(raw_files)} .raw files", flush=True)

    all_feat, all_labels, all_ids = [], [], []

    for fpath in raw_files:
        fname = fpath.name
        if fname.startswith("0201"):
            label = 1  # MDD
        elif fname.startswith("0202") or fname.startswith("0203"):
            label = 0  # HC
        else:
            continue

        m = re.match(r'(\d{8})', fname)
        sub_id = m.group(1) if m else fname.split("erp")[0].strip().replace("_", "")

        try:
            raw = mne.io.read_raw_egi(str(fpath), preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_EEG_CH])
            raw.filter(FMIN, FMAX, verbose=False)

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                print(f"  Skip {fname}: condition '{CONDITION}' not found", flush=True)
                del raw; gc.collect(); continue

            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(
                raw, cond_events, tmin=TMIN, tmax=TMAX,
                baseline=BASELINE, preload=True, verbose=False,
                reject=dict(eeg=150e-6)
            )
            data = epochs.get_data()   # (n_trials, 128, n_times)
            times = epochs.times
            del raw, epochs; gc.collect()

            if data.shape[0] < 10:
                print(f"  Skip {fname}: only {data.shape[0]} trials (<10)", flush=True)
                del data; continue

            n_trials = data.shape[0]
            # Average ERP across trials, extract mean amplitude in P300 window
            avg_erp = data.mean(axis=0)                          # (128, n_times)
            p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
            feat = avg_erp[:, p300_mask].mean(axis=1)            # (128,)
            del data, avg_erp

            all_feat.append(feat)
            all_labels.append(label)
            all_ids.append(sub_id)
            print(f"  {sub_id}: label={label}, trials={n_trials}", flush=True)

        except Exception as e:
            print(f"  Skip {fname}: {e}", flush=True)
            gc.collect()

    X = np.array(all_feat)    # (n_subjects, 128)
    y = np.array(all_labels)
    ids = np.array(all_ids)
    n_mdd = sum(y == 1); n_hc = sum(y == 0)
    print(f"\nTotal: {len(y)} subjects (MDD={n_mdd}, HC={n_hc})", flush=True)
    return X, y, ids


def load_erp_timeseries(condition="hcue"):
    """One avg-ERP timeseries per subject; also returns P300 feature vector."""
    global CONDITION
    CONDITION = condition
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    all_feat, all_erps, all_labels, all_ids = [], [], [], []
    times_ref = None

    for fpath in raw_files:
        fname = fpath.name
        if fname.startswith("0201"):
            label = 1
        elif fname.startswith("0202") or fname.startswith("0203"):
            label = 0
        else:
            continue

        m = re.match(r'(\d{8})', fname)
        sub_id = m.group(1) if m else fname.split("erp")[0].strip().replace("_", "")

        try:
            raw = mne.io.read_raw_egi(str(fpath), preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_EEG_CH])
            raw.filter(FMIN, FMAX, verbose=False)

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                del raw; gc.collect(); continue

            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(
                raw, cond_events, tmin=TMIN, tmax=TMAX,
                baseline=BASELINE, preload=True, verbose=False,
                reject=dict(eeg=150e-6)
            )
            data = epochs.get_data()   # (n_trials, 128, n_times)
            if times_ref is None:
                times_ref = epochs.times.copy()
            del raw, epochs; gc.collect()

            if data.shape[0] < 10:
                del data; continue

            avg_erp = data.mean(axis=0)   # (128, n_times)
            assert avg_erp.shape[1] == len(times_ref), "times mismatch"
            p300_mask = (times_ref >= P300_WIN[0]) & (times_ref <= P300_WIN[1])
            feat = avg_erp[:, p300_mask].mean(axis=1)
            del data

            all_feat.append(feat)
            all_erps.append(avg_erp)
            all_labels.append(label)
            all_ids.append(sub_id)

        except Exception as e:
            print(f"  Skip {fname}: {e}", flush=True)
            gc.collect()

    return (np.array(all_feat), np.array(all_erps),
            times_ref, np.array(all_labels), np.array(all_ids))


def load_erp_features_dict(condition):
    """Load features as dict[sub_id -> (feat, label)] for multi-condition fusion."""
    global CONDITION
    CONDITION = condition
    X, y, ids = load_erp_features()
    return {sid: (X[i], y[i]) for i, sid in enumerate(ids)}


def fuse_conditions(cond_dicts):
    """Intersect sub_ids across conditions and hstack features."""
    common = sorted(set.intersection(*[set(d.keys()) for d in cond_dicts]))
    X = np.hstack([np.array([d[s][0] for s in common]) for d in cond_dicts])
    y = np.array([cond_dicts[0][s][1] for s in common])
    ids = np.array(common)
    return X, y, ids


def _loso_ba_subset(X, y, C=1.0):
    """LOSO BA for low-dim subsets; PCA n_components capped at feature count."""
    nc = min(X.shape[1], len(y) - 1)
    loo = LeaveOneOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=nc)),
        ("clf",    LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    ])
    preds = []
    for tr_idx, te_idx in loo.split(X):
        pipe.fit(X[tr_idx], y[tr_idx])
        preds.append(int(pipe.predict(X[te_idx])[0]))
    return balanced_accuracy_score(y, preds)


def permutation_test_subset(X, y, n_perm=1000, seed=42, C=1.0):
    """Permutation test using _loso_ba_subset."""
    assert len(y) == X.shape[0]
    rng = np.random.default_rng(seed)
    obs_ba = _loso_ba_subset(X, y, C=C)
    seeds = rng.integers(0, 2**31, size=n_perm)

    def one_perm(s):
        return _loso_ba_subset(X, np.random.default_rng(s).permutation(y), C=C)

    print(f"Running {n_perm} permutations (n_jobs=4)...", flush=True)
    perm_bas = Parallel(n_jobs=4)(delayed(one_perm)(s) for s in seeds)
    perm_arr = np.array(perm_bas)
    p_val = float((np.sum(perm_arr >= obs_ba) + 1) / (n_perm + 1))
    print(f"  Observed BA={obs_ba:.3f}  p={p_val:.4f}  "
          f"(perm mean={np.mean(perm_arr):.3f})", flush=True)
    return obs_ba, p_val


def run_electrode_selection(X, y, ids, ba_baseline=None, p_baseline=None):
    """Phase 10: compare 3-dim and 5-dim parietal subsets vs 128-dim baseline."""
    print("\n" + "="*50)
    print("Phase 10: Electrode Selection")
    print("="*50)
    print("\nEGI HydroCel-128 parietal channel indices:")
    idx = get_parietal_indices()

    ch3 = ["Pz", "P3", "P4"]
    X_3 = X[:, [idx[c] for c in ch3]]
    print(f"\n3-dim {ch3}  shape={X_3.shape}")
    ba3, p3 = permutation_test_subset(X_3, y, n_perm=1000)

    ch5 = ["Pz", "P3", "P4", "Cz", "CPz"]
    # deduplicate indices while preserving order
    seen, unique_cols = set(), []
    for c in ch5:
        i = idx[c]
        if i not in seen:
            seen.add(i)
            unique_cols.append(i)
    X_5 = X[:, unique_cols]
    n_unique = len(unique_cols)
    print(f"\n{n_unique}-dim {ch5} (unique cols={n_unique})  shape={X_5.shape}")
    ba5, p5 = permutation_test_subset(X_5, y, n_perm=1000)

    ba_b = ba_baseline if ba_baseline is not None else 0.670
    p_b  = p_baseline  if p_baseline  is not None else 0.021
    print("\n" + "="*50)
    print("Electrode Selection Results vs Baseline:")
    print(f"  128-dim (baseline): BA={ba_b:.3f}  p={p_b:.4f}")
    print(f"  3-dim  (Pz/P3/P4): BA={ba3:.3f}  p={p3:.4f}")
    print(f"  {n_unique}-dim  (+Cz/CPz):  BA={ba5:.3f}  p={p5:.4f}")
    print("="*50)


def _loso_ba(X, y):
    """Silent LOSO — returns BA only (for permutation loop)."""
    loo = LeaveOneOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=20)),
        ("clf",    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    ])
    preds = []
    for tr_idx, te_idx in loo.split(X):
        pipe.fit(X[tr_idx], y[tr_idx])
        preds.append(int(pipe.predict(X[te_idx])[0]))
    return balanced_accuracy_score(y, preds)


def permutation_test_loso(X, y, n_perm=1000, seed=42):
    """Subject-level permutation test. Features extracted once outside loop."""
    assert len(y) == X.shape[0], "y must be subject-level"
    rng = np.random.default_rng(seed)
    obs_ba = _loso_ba(X, y)
    seeds = rng.integers(0, 2**31, size=n_perm)

    def one_perm(s):
        return _loso_ba(X, np.random.default_rng(s).permutation(y))

    print(f"Running {n_perm} permutations (n_jobs=4)...", flush=True)
    perm_bas = Parallel(n_jobs=4)(delayed(one_perm)(s) for s in seeds)
    perm_arr = np.array(perm_bas)
    p_val = float((np.sum(perm_arr >= obs_ba) + 1) / (n_perm + 1))
    print(f"  Observed BA={obs_ba:.3f}  p={p_val:.4f}  "
          f"(perm mean={np.mean(perm_arr):.3f})", flush=True)
    return obs_ba, p_val


def run_loso(X, y, ids, label=None):
    """Leave-one-subject-out on subject-level features."""
    loo = LeaveOneOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=20)),
        ("clf",    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    ])

    true_labels, probs = [], []
    n = len(y)
    print(f"\nRunning LOSO: {n} folds...", flush=True)

    for i, (tr_idx, te_idx) in enumerate(loo.split(X)):
        pipe.fit(X[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(X[te_idx])[:, 1][0]
        true_labels.append(y[te_idx][0])
        probs.append(prob)

        pred = int(prob >= 0.5)
        status = "OK" if pred == y[te_idx][0] else "MISS"
        print(f"  {i+1}/{n}: {ids[te_idx][0]} true={y[te_idx][0]} "
              f"prob={prob:.3f} [{status}]", flush=True)

    true_labels = np.array(true_labels)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)

    ba  = balanced_accuracy_score(true_labels, preds)
    auc = roc_auc_score(true_labels, probs)

    print(f"\n{'='*50}")
    print(f"Subject-level results ({label or CONDITION} condition, avg-ERP P300):")
    print(f"  BA  = {ba:.3f}")
    print(f"  AUC = {auc:.3f}")
    print(f"  Acc = {np.mean(preds == true_labels):.3f}")
    print(f"  N   = {len(true_labels)} (MDD={sum(true_labels==1)}, HC={sum(true_labels==0)})")
    print(f"{'='*50}")
    return ba, auc


def plot_ttest_curve(avg_erps, times, y, out_dir):
    """TWIN-01: per-timepoint t-test curve with FDR correction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parietal_idx = list(dict.fromkeys(get_parietal_indices().values()))
    mask_time = (times >= 0.25) & (times <= 0.50)
    t_times = times[mask_time]
    erp_par = avg_erps[:, parietal_idx, :][:, :, mask_time].mean(axis=1)  # (N, n_t)

    mdd_mask = y == 1
    t_stats, p_vals = zip(*[ttest_ind(erp_par[mdd_mask, i], erp_par[~mdd_mask, i],
                                      equal_var=False) for i in range(erp_par.shape[1])])
    t_stats = np.array(t_stats)
    p_vals = np.array(p_vals)
    adj_p = false_discovery_control(p_vals)
    sig_mask = adj_p < 0.05
    peak_t = t_times[np.argmax(np.abs(t_stats))]

    fig, ax = plt.subplots(figsize=(8, 4))
    bin_colors = ["#e8e8e8", "#d8d8d8", "#e8e8e8", "#d8d8d8", "#e8e8e8"]
    for (label, t0, t1), col in zip(BINS, bin_colors):
        ax.axvspan(t0, t1, alpha=0.15, color=col)
    ax.plot(t_times, t_stats, color="steelblue", lw=1.5)
    if sig_mask.any():
        ax.scatter(t_times[sig_mask], t_stats[sig_mask], color="red", s=20, zorder=5,
                   label=f"FDR sig (n={sig_mask.sum()})")
    ax.axvline(peak_t, color="orange", lw=1.5, linestyle="--",
               label=f"Peak |t| @ {peak_t*1000:.0f}ms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("t-statistic")
    ax.set_title("Per-timepoint t-test (MDD vs HC, parietal avg)")
    ax.legend(fontsize=8)
    out_path = Path(out_dir) / "ttest_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Peak |t| at {peak_t*1000:.1f}ms  FDR-sig points: {sig_mask.sum()}", flush=True)
    print(f"  Saved: {out_path}", flush=True)


def run_bin_ablation(avg_erps, times, y, out_dir):
    """TWIN-02: 50ms bin ablation classification."""
    rows = []
    best_ba = -1
    for name, t0, t1 in BINS:
        if name == BINS[-1][0]:
            mask = (times >= t0) & (times <= t1)
        else:
            mask = (times >= t0) & (times < t1)
        X_bin = avg_erps[:, :, mask].mean(axis=2)  # (N, 128)
        ba, p = permutation_test_subset(X_bin, y, n_perm=1000, C=0.1)
        rows.append((name, ba, p))
        if ba > best_ba:
            best_ba = ba

    print(f"\n{'Bin':<12} {'BA':>6} {'p-value':>8} {'Sig':>4} {'Note':>6}", flush=True)
    print("-" * 42, flush=True)
    for name, ba, p in rows:
        sig = "*" if p < 0.05 else ""
        note = "best" if ba == best_ba else ""
        print(f"{name:<12} {ba:>6.3f} {p:>8.4f} {sig:>4} {note:>6}", flush=True)

    out_path = Path(out_dir) / "bin_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Bin", "BA", "p-value", "Sig", "Note"])
        for name, ba, p in rows:
            sig = "*" if p < 0.05 else ""
            note = "best" if ba == best_ba else ""
            writer.writerow([name, f"{ba:.3f}", f"{p:.4f}", sig, note])
    print(f"  Saved: {out_path}", flush=True)


def run_time_window_analysis():
    print("\n" + "="*50, flush=True)
    print("Phase 11: Time-Window Analysis", flush=True)
    print("="*50, flush=True)
    out_dir = Path("out_phase11")
    out_dir.mkdir(exist_ok=True)
    _, avg_erps, times, y, _ = load_erp_timeseries(condition="hcue")
    print(f"  avg_erps shape: {avg_erps.shape}", flush=True)
    print("\nTWIN-01: Per-timepoint t-test curve", flush=True)
    plot_ttest_curve(avg_erps, times, y, out_dir)
    print("\nTWIN-02: 50ms bin ablation (128-dim, C=0.1)", flush=True)
    run_bin_ablation(avg_erps, times, y, out_dir)


def _loso_collect_coefs(X, y, n_components=20, C=1.0):
    """LOSO returning per-fold back-projected channel weights (n_folds, 128)."""
    loo = LeaveOneOut()
    fold_weights = []
    for tr_idx, _ in loo.split(X):
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        X_pca = pca.fit_transform(scaler.fit_transform(X[tr_idx]))
        clf.fit(X_pca, y[tr_idx])
        fold_weights.append(pca.components_.T @ clf.coef_[0])  # (128,)
    return np.array(fold_weights)  # (n_folds, 128)


def run_feature_importance():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "="*50, flush=True)
    print("Phase 12: Feature Importance", flush=True)
    print("="*50, flush=True)

    out_dir = Path("out_phase12")
    out_dir.mkdir(exist_ok=True)

    # Load hcue full-window features
    global CONDITION
    CONDITION = "hcue"
    X_hcue, y, _ = load_erp_features()

    # Load bin250 features (250-300ms mean per channel)
    _, avg_erps, times, y_ts, _ = load_erp_timeseries(condition="hcue")
    mask = (times >= 0.250) & (times < 0.300)
    X_bin250 = avg_erps[:, :, mask].mean(axis=2)  # (n_subjects, 128)

    # Build MNE Info once
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    info = mne.create_info(montage.ch_names[:128], sfreq=1000, ch_types="eeg")
    info.set_montage(montage)
    ch_names = montage.ch_names[:128]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, (label, C_val, X_model, y_model) in enumerate([
            ("hcue", 1.0, X_hcue, y),
            ("bin250", 0.1, X_bin250, y_ts)]):

        print(f"\nCollecting LOSO coefs for {label} (C={C_val})...", flush=True)
        fold_weights = _loso_collect_coefs(X_model, y_model, n_components=20, C=C_val)
        avg_weights = fold_weights.mean(axis=0)
        avg_abs = np.abs(fold_weights).mean(axis=0)
        rank_idx = np.argsort(avg_abs)[::-1]

        # Print top-10 table
        print(f"\nFeature Importance: {label}", flush=True)
        print(f"{'Rank':>4}  {'Channel':<10}  {'Avg|weight|':>11}  {'Sign':<6}", flush=True)
        print("-" * 38, flush=True)
        rows = []
        for i in range(128):
            ri = rank_idx[i]
            sign = "MDD+" if avg_weights[ri] > 0 else "HC+"
            rows.append({"rank": i+1, "channel": ch_names[ri],
                         "avg_abs_weight": float(avg_abs[ri]),
                         "avg_weight": float(avg_weights[ri]),
                         "sign": sign})
            if i < 10:
                print(f"{i+1:>4}  {ch_names[ri]:<10}  {avg_abs[ri]:>11.6f}  {sign:<6}", flush=True)

        # Save CSV
        csv_path = out_dir / f"feature_importance_{label}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}", flush=True)

        # Topomap
        names_list = [""] * 128
        for i in range(5):
            names_list[rank_idx[i]] = ch_names[rank_idx[i]]
        vmax = float(np.abs(avg_weights).max())
        im, _ = mne.viz.plot_topomap(avg_weights, info, cmap="RdBu_r",
                                     vlim=(-vmax, vmax), names=names_list,
                                     axes=axes[idx], show=False)
        axes[idx].set_title(f"Feature Importance: {label}")
        plt.colorbar(im, ax=axes[idx], shrink=0.7)

    fig.savefig(out_dir / "topomap_hcue_vs_bin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/topomap_hcue_vs_bin.png", flush=True)


def load_erp_features_latency():
    """P300 amplitude + P300 peak latency per channel → (n_subjects, 256)."""
    global CONDITION
    CONDITION = "hcue"
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    all_feat, all_labels, all_ids = [], [], []

    for fpath in raw_files:
        fname = fpath.name
        if fname.startswith("0201"):
            label = 1
        elif fname.startswith("0202") or fname.startswith("0203"):
            label = 0
        else:
            continue
        m = re.match(r'(\d{8})', fname)
        sub_id = m.group(1) if m else fname.split("erp")[0].strip().replace("_", "")
        try:
            raw = mne.io.read_raw_egi(str(fpath), preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_EEG_CH])
            raw.filter(FMIN, FMAX, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                del raw; gc.collect(); continue
            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(raw, cond_events, tmin=TMIN, tmax=TMAX,
                                baseline=BASELINE, preload=True, verbose=False,
                                reject=dict(eeg=150e-6))
            data = epochs.get_data()
            times = epochs.times
            del raw, epochs; gc.collect()
            if data.shape[0] < 10:
                del data; continue
            avg_erp = data.mean(axis=0)          # (128, n_times)
            p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
            p300_seg = avg_erp[:, p300_mask]     # (128, n_p300)
            p300_times = times[p300_mask]
            amp = p300_seg.mean(axis=1)          # (128,)
            lat = p300_times[np.argmax(p300_seg, axis=1)]  # (128,) peak latency
            feat = np.concatenate([amp, lat])    # (256,)
            del data, avg_erp
            all_feat.append(feat)
            all_labels.append(label)
            all_ids.append(sub_id)
        except Exception as e:
            print(f"  Skip {fname}: {e}", flush=True)
            gc.collect()

    X = np.array(all_feat)
    y = np.array(all_labels)
    ids = np.array(all_ids)
    print(f"Latency features: {X.shape}", flush=True)
    return X, y, ids


def load_erp_features_extended():
    """P300 + N200 mean amplitude per channel → (n_subjects, 256)."""
    global CONDITION
    CONDITION = "hcue"
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    all_feat, all_labels, all_ids = [], [], []

    for fpath in raw_files:
        fname = fpath.name
        if fname.startswith("0201"):
            label = 1
        elif fname.startswith("0202") or fname.startswith("0203"):
            label = 0
        else:
            continue
        m = re.match(r'(\d{8})', fname)
        sub_id = m.group(1) if m else fname.split("erp")[0].strip().replace("_", "")
        try:
            raw = mne.io.read_raw_egi(str(fpath), preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_EEG_CH])
            raw.filter(FMIN, FMAX, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                del raw; gc.collect(); continue
            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(raw, cond_events, tmin=TMIN, tmax=TMAX,
                                baseline=BASELINE, preload=True, verbose=False,
                                reject=dict(eeg=150e-6))
            data = epochs.get_data()
            times = epochs.times
            del raw, epochs; gc.collect()
            if data.shape[0] < 10:
                del data; continue
            avg_erp = data.mean(axis=0)
            p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
            n200_mask = (times >= N200_WIN[0]) & (times <= N200_WIN[1])
            feat = np.concatenate([
                avg_erp[:, p300_mask].mean(axis=1),   # (128,)
                avg_erp[:, n200_mask].mean(axis=1),   # (128,)
            ])
            del data, avg_erp
            all_feat.append(feat)
            all_labels.append(label)
            all_ids.append(sub_id)
        except Exception as e:
            print(f"  Skip {fname}: {e}", flush=True)
            gc.collect()

    X = np.array(all_feat)
    y = np.array(all_labels)
    ids = np.array(all_ids)
    print(f"Extended features: {X.shape}", flush=True)
    return X, y, ids


def _loso_ba_clf(X, y, clf):
    """LOSO BA with arbitrary sklearn classifier (wrapped in scaler+pca pipeline)."""
    n_comp = min(20, X.shape[1], X.shape[0] - 1)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_comp)),
        ("clf",    clf),
    ])
    loo = LeaveOneOut()
    preds = []
    for tr_idx, te_idx in loo.split(X):
        pipe.fit(X[tr_idx], y[tr_idx])
        preds.append(int(pipe.predict(X[te_idx])[0]))
    return balanced_accuracy_score(y, preds)


def run_classifier_comparison():
    """Phase 14: compare LR on P300 / P300+latency / P300+N200 features."""
    print("\n" + "="*50)
    print("Classifier Comparison (LOSO)")
    print("="*50)

    import copy
    CONDITION = "hcue"
    X_p300, y, _ = load_erp_features()
    X_lat,  _, _ = load_erp_features_latency()
    X_ext,  _, _ = load_erp_features_extended()

    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")

    print(f"\n{'Features':<22} BA")
    print("-" * 30)
    for label, X in [("P300-amp(128)", X_p300),
                     ("P300-amp+lat(256)", X_lat),
                     ("P300+N200-amp(256)", X_ext)]:
        ba = _loso_ba_clf(X, y, copy.deepcopy(lr))
        print(f"{label:<22} {ba:.3f}", flush=True)


def load_single_trial_features():
    """One feature vector per trial: mean P300 amplitude per channel.
    Returns X (n_trials, 128), y (n_trials,), groups (n_trials,) subject index.
    """
    global CONDITION
    CONDITION = "hcue"
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    all_feat, all_labels, all_groups = [], [], []
    sub_idx = 0

    for fpath in raw_files:
        fname = fpath.name
        if fname.startswith("0201"):
            label = 1
        elif fname.startswith("0202") or fname.startswith("0203"):
            label = 0
        else:
            continue
        try:
            raw = mne.io.read_raw_egi(str(fpath), preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_EEG_CH])
            raw.filter(FMIN, FMAX, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                del raw; gc.collect(); continue
            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(raw, cond_events, tmin=TMIN, tmax=TMAX,
                                baseline=BASELINE, preload=True, verbose=False,
                                reject=dict(eeg=150e-6))
            data = epochs.get_data()   # (n_trials, 128, n_times)
            times = epochs.times
            del raw, epochs; gc.collect()
            if data.shape[0] < 10:
                del data; continue
            p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
            feats = data[:, :, p300_mask].mean(axis=2)  # (n_trials, 128)
            n_trials = feats.shape[0]
            all_feat.append(feats)
            all_labels.extend([label] * n_trials)
            all_groups.extend([sub_idx] * n_trials)
            sub_idx += 1
            del data
        except Exception as e:
            print(f"  Skip {fname}: {e}", flush=True)
            gc.collect()

    X = np.vstack(all_feat)
    y = np.array(all_labels)
    groups = np.array(all_groups)
    print(f"Single-trial: {X.shape[0]} trials, {sub_idx} subjects", flush=True)
    return X, y, groups


def run_single_trial_analysis():
    """Phase 13: single-trial LOSO classification."""
    print("\n" + "="*50)
    print("Single-Trial Classification (LeaveOneGroupOut)")
    print("="*50)
    X, y, groups = load_single_trial_features()

    logo = LeaveOneGroupOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=20)),
        ("clf",    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    ])
    preds, trues = [], []
    for tr_idx, te_idx in logo.split(X, y, groups):
        pipe.fit(X[tr_idx], y[tr_idx])
        preds.extend(pipe.predict(X[te_idx]).tolist())
        trues.extend(y[te_idx].tolist())
    ba = balanced_accuracy_score(trues, preds)
    print(f"  BA={ba:.3f}  (n_trials={len(trues)})", flush=True)
    return ba


def run_subsample_ensemble(n_bags=10, k_trials=15, seed=42):
    """Approach 1: sub-sample averaging + ensemble bagging."""
    print("\n" + "="*50)
    print("Approach 1: Sub-sample Averaging + Ensemble")
    print(f"  n_bags={n_bags}, k_trials={k_trials}")
    print("="*50)
    global CONDITION; CONDITION = "hcue"
    rng = np.random.default_rng(seed)

    # Load raw trial data per subject
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    subjects = []  # list of (trial_data, label, sub_id)
    for fpath in raw_files:
        fname = fpath.name
        if fname.startswith("0201"): label = 1
        elif fname.startswith("0202") or fname.startswith("0203"): label = 0
        else: continue
        m = re.match(r'(\d{8})', fname)
        sub_id = m.group(1) if m else fname
        try:
            raw = mne.io.read_raw_egi(str(fpath), preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_EEG_CH])
            raw.filter(FMIN, FMAX, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                del raw; gc.collect(); continue
            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(raw, cond_events, tmin=TMIN, tmax=TMAX,
                                baseline=BASELINE, preload=True, verbose=False,
                                reject=dict(eeg=150e-6))
            data = epochs.get_data()
            times = epochs.times
            del raw, epochs; gc.collect()
            if data.shape[0] < k_trials:
                del data; continue
            subjects.append((data, times, label, sub_id))
            print(f"  {sub_id}: label={label}, trials={data.shape[0]}", flush=True)
        except Exception as e:
            print(f"  Skip {fname}: {e}", flush=True); gc.collect()

    n_sub = len(subjects)
    y_sub = np.array([s[2] for s in subjects])
    print(f"  Loaded {n_sub} subjects (MDD={sum(y_sub==1)}, HC={sum(y_sub==0)})")

    # Build bagged pseudo-averages per subject
    def make_bags(trial_data, times):
        p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
        bags = []
        for _ in range(n_bags):
            idx = rng.choice(trial_data.shape[0], size=k_trials, replace=True)
            avg = trial_data[idx].mean(axis=0)  # (128, n_times)
            bags.append(avg[:, p300_mask].mean(axis=1))  # (128,)
        return np.array(bags)  # (n_bags, 128)

    all_bags = [make_bags(s[0], s[1]) for s in subjects]

    # LOSO with ensemble: train on all bags from other subjects, predict held-out
    loo = LeaveOneOut()
    preds = []
    for tr_idx, te_idx in loo.split(y_sub):
        X_tr = np.vstack([all_bags[i] for i in tr_idx])
        y_tr = np.repeat(y_sub[tr_idx], n_bags)
        X_te = all_bags[te_idx[0]]  # (n_bags, 128)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(20, X_tr.shape[0]-1))),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        ])
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1].mean()
        preds.append(int(prob >= 0.5))

    ba = balanced_accuracy_score(y_sub, preds)
    print(f"\n  Ensemble BA={ba:.3f}  (ERP baseline=0.670, delta={ba-0.670:+.3f})")
    return ba


def run_roi_timebin_features():
    """Approach 2: low-dim ROI × time-bin features."""
    print("\n" + "="*50)
    print("Approach 2: Low-dim ROI x Time-bin Features")
    print("="*50)

    _, avg_erps, times, y, ids = load_erp_timeseries(condition="hcue")
    parietal_idx = list(dict.fromkeys(get_parietal_indices(("Pz","P3","P4")).values()))
    central_idx = list(dict.fromkeys(get_parietal_indices(("Cz","CPz")).values()))
    frontal_idx = list(dict.fromkeys(get_parietal_indices(("Fz",)).values()))

    rois = [("parietal", parietal_idx), ("central", central_idx), ("frontal", frontal_idx)]
    feats = []
    feat_names = []
    for roi_name, ch_idx in rois:
        roi_erp = avg_erps[:, ch_idx, :].mean(axis=1)  # (N, n_times)
        for bin_name, t0, t1 in BINS:
            mask = (times >= t0) & (times < t1) if bin_name != BINS[-1][0] else (times >= t0) & (times <= t1)
            feats.append(roi_erp[:, mask].mean(axis=1, keepdims=True))
            feat_names.append(f"{roi_name}_{bin_name}")

    X = np.hstack(feats)  # (N, n_rois * n_bins)
    print(f"  Feature matrix: {X.shape}  ({len(feat_names)} dims)")
    print(f"  Features: {feat_names}")

    # LOSO with regularized LR (no PCA needed — already low-dim)
    loo = LeaveOneOut()
    preds_lr, preds_svm = [], []
    for tr_idx, te_idx in loo.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
        lr.fit(X_tr, y[tr_idx])
        preds_lr.append(int(lr.predict(X_te)[0]))
        from sklearn.svm import LinearSVC
        svm = LinearSVC(C=0.1, max_iter=5000, class_weight="balanced")
        svm.fit(X_tr, y[tr_idx])
        preds_svm.append(int(svm.predict(X_te)[0]))

    ba_lr = balanced_accuracy_score(y, preds_lr)
    ba_svm = balanced_accuracy_score(y, preds_svm)
    print(f"\n  LR  BA={ba_lr:.3f}  (delta={ba_lr-0.670:+.3f})")
    print(f"  SVM BA={ba_svm:.3f}  (delta={ba_svm-0.670:+.3f})")
    print(f"  ERP baseline: BA=0.670")
    return ba_lr, ba_svm


def load_resting_features():
    from scipy.io import loadmat
    from scipy.signal import welch
    X, y, ids = [], [], []
    for f in sorted(RESTING_DIR.glob("*.mat")):
        sid = f.stem
        try:
            mat = loadmat(str(f))
            data_key = next(k for k in mat if not k.startswith("_") and k not in ("samplingRate", "Impedances_0"))
            data = mat[data_key][:128, :]
            fs = float(mat["samplingRate"].flat[0])
            freqs, psd = welch(data, fs=fs, nperseg=256)
            feats = []
            for lo, hi in BANDS.values():
                mask = (freqs >= lo) & (freqs < hi)
                feats.append(psd[:, mask].mean(axis=1))
            X.append(np.concatenate(feats))
            label = 1 if sid.startswith("0201") else 0
            y.append(label); ids.append(sid)
            print(f"  {sid}: label={label}", flush=True)
        except Exception as e:
            print(f"  Skip {f.name}: {e}", flush=True)
    X, y = np.array(X), np.array(y)
    print(f"Resting: {X.shape}  MDD={sum(y==1)} HC={sum(y==0)}", flush=True)
    return X, y, ids


def run_resting_analysis():
    print("\n" + "="*50)
    print("Phase 13: Resting-State EEG Classification")
    print("="*50)
    out_dir = Path("out_phase13"); out_dir.mkdir(exist_ok=True)
    X, y, _ = load_resting_features()
    n = len(y)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=min(20, n - 1))),
        ("clf",    LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])
    loo = LeaveOneOut()
    preds, probs = np.zeros(n, dtype=int), np.zeros(n)
    for tr, te in loo.split(X):
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
        probs[te]  = pipe.predict_proba(X[te])[:, 1]
    ba  = balanced_accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    lines = [
        f"Resting-state EEG classification (N={n})",
        f"BA={ba:.3f}  AUC={auc:.3f}",
        f"ERP baseline: BA=0.670",
        f"Delta BA: {ba - 0.670:+.3f}",
    ]
    print("\n".join(lines), flush=True)
    (out_dir / "resting_result.txt").write_text("\n".join(lines))
    print(f"  Saved: out_phase13/resting_result.txt", flush=True)


if __name__ == "__main__":
    # Phase 8: hcue single-condition + permutation test
    CONDITION = "hcue"
    X, y, ids = load_erp_features()
    print(f"Feature matrix: {X.shape}", flush=True)
    run_loso(X, y, ids, label="hcue")
    permutation_test_loso(X, y, n_perm=1000)

    # Phase 9: multi-condition fusion (hcue + fcue + scue)
    print("\n" + "="*50)
    print("Multi-condition fusion: hcue + fcue + scue")
    dicts = [load_erp_features_dict(c) for c in ["hcue", "fcue", "scue"]]
    X_fused, y_fused, ids_fused = fuse_conditions(dicts)
    print(f"Fused matrix: {X_fused.shape}  (n={len(y_fused)})", flush=True)
    run_loso(X_fused, y_fused, ids_fused, label="fusion(hcue+fcue+scue)")
    permutation_test_loso(X_fused, y_fused, n_perm=1000)

    # Contrast: scue - hcue
    print("\n" + "="*50)
    print("Contrast: scue - hcue")
    common = sorted(set(dicts[0]) & set(dicts[2]))
    X_contrast = np.array([dicts[2][s][0] - dicts[0][s][0] for s in common])
    y_contrast = np.array([dicts[0][s][1] for s in common])
    ids_contrast = np.array(common)
    print(f"Contrast matrix: {X_contrast.shape}", flush=True)
    run_loso(X_contrast, y_contrast, ids_contrast, label="contrast(scue-hcue)")
    permutation_test_loso(X_contrast, y_contrast, n_perm=1000)

    # Phase 10: Electrode Selection (reuse hcue X, y, ids from top of __main__)
    ba_base, p_base = permutation_test_loso(X, y, n_perm=1000)
    run_electrode_selection(X, y, ids, ba_baseline=ba_base, p_baseline=p_base)

    # Phase 11: Time-Window Analysis
    run_time_window_analysis()

    # Phase 12: Feature Importance
    run_feature_importance()

    # Phase 13: Single-Trial Classification
    run_single_trial_analysis()

    # Phase 14: Classifier Comparison (LR / SVM / XGBoost × P300 / P300+N200)
    run_classifier_comparison()

    # Phase 13 (Resting-State): band-power LOSO vs ERP baseline
    run_resting_analysis()
