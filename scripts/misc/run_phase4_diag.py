"""Phase 4 diagnostic: test feature variants to find BA>0.60."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

import numpy as np, logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from modma_mdd_real_experiment import (
    build_region_indices, compute_subject_balanced_sample_weights,
    _alpha_asymmetry, EGI128_LEFT, EGI128_RIGHT, REGIONS,
)
from run_diagnostic import load_filtered_data
from scipy.signal import welch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
SFREQ = 125.0


def make_features(X, ch_names, config):
    """Build feature matrix from config dict."""
    n_win, n_ch, n_times = X.shape
    ri = build_region_indices(ch_names)
    nperseg = min(n_times, int(SFREQ * 2))
    freqs, psd = welch(X, fs=SFREQ, axis=2, nperseg=nperseg)
    tp = np.sum(psd, axis=2)

    def brel(region, fmin, fmax):
        idx = ri[region]
        m = (freqs >= fmin) & (freqs <= fmax)
        return (np.mean(np.mean(psd[:, idx][:, :, m], axis=2), axis=1) /
                (np.mean(tp[:, idx], axis=1) + 1e-10))[:, None]

    def tbr(region):
        idx = ri[region]
        tm = (freqs >= 4) & (freqs <= 8)
        bm = (freqs >= 13) & (freqs <= 30)
        t = np.mean(np.mean(psd[:, idx][:, :, tm], axis=2), axis=1, keepdims=True)
        b = np.mean(np.mean(psd[:, idx][:, :, bm], axis=2), axis=1, keepdims=True)
        return t / (b + 1e-10)

    def asym(region):
        am = (freqs >= 8) & (freqs <= 13)
        ap = np.mean(psd[:, :, am], axis=2)
        return _alpha_asymmetry(ap, ch_names, n_win, region)

    def riem(r1, r2):
        s = [np.mean(X[:, ri[r], :], axis=1) if ri[r] else np.zeros((n_win, n_times))
             for r in [r1, r2]]
        s = np.stack(s, axis=1)
        c = np.array([np.cov(s[i]) for i in range(n_win)])
        c += 1e-6 * np.eye(2)[None]
        return np.log(np.abs(c[:, 0, 1:2]) + 1e-10)

    def hjorth_act(region):
        idx = ri[region]
        sig = np.mean(X[:, idx, :], axis=1) if idx else np.zeros((n_win, n_times))
        return np.var(sig, axis=1, keepdims=True)

    feats, names = [], []
    for item in config:
        t, args = item[0], item[1:]
        if t == "brel":
            feats.append(brel(*args)); names.append(f"{args[0]}_{args[1]}_{args[2]}_rel")
        elif t == "tbr":
            feats.append(tbr(args[0])); names.append(f"{args[0]}_TBR")
        elif t == "asym":
            feats.append(asym(args[0])); names.append(f"asym_{args[0]}")
        elif t == "riem":
            feats.append(riem(*args)); names.append(f"riem_{args[0]}_{args[1]}")
        elif t == "hjorth":
            feats.append(hjorth_act(args[0])); names.append(f"hjorth_{args[0]}")
    return np.column_stack(feats), names


def run_cv(feats, y, groups):
    ug, fi = np.unique(groups, return_index=True)
    gl = y[fi]
    ns = max(2, min(5, np.min(np.unique(gl, return_counts=True)[1])))
    cv = StratifiedGroupKFold(n_splits=ns)
    yt_s, yp_s = {}, {}
    for tr, te in cv.split(feats, y, groups=groups):
        pipe = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(C=0.1, class_weight="balanced",
                                                  max_iter=1000, random_state=42))])
        sw = compute_subject_balanced_sample_weights(groups[tr])
        pipe.fit(feats[tr], y[tr], c__sample_weight=sw)
        p = pipe.predict_proba(feats[te])[:, 1]
        for i, g in enumerate(groups[te]):
            if g not in yt_s: yt_s[g] = y[te][i]; yp_s[g] = []
            yp_s[g].append(p[i])
    subjs = list(yt_s.keys())
    yt = np.array([yt_s[s] for s in subjs])
    yp = (np.array([np.mean(yp_s[s]) for s in subjs]) >= 0.5).astype(int)
    return balanced_accuracy_score(yt, yp)


CONFIGS = {
    # Baseline: current 5-dim
    "A_baseline_5dim": [
        ("brel", "frontal", 4, 8), ("brel", "frontal", 8, 13),
        ("asym", "frontal"), ("tbr", "frontal"), ("riem", "central", "temporal"),
    ],
    # B: swap frontal→parietal for theta/alpha
    "B_parietal_focus": [
        ("brel", "parietal", 4, 8), ("brel", "parietal", 8, 13),
        ("asym", "frontal"), ("tbr", "parietal"), ("riem", "central", "temporal"),
    ],
    # C: add hjorth_frontal + hjorth_central
    "C_add_hjorth": [
        ("brel", "frontal", 4, 8), ("brel", "frontal", 8, 13),
        ("asym", "frontal"), ("tbr", "frontal"), ("riem", "central", "temporal"),
        ("hjorth", "frontal"), ("hjorth", "central"),
    ],
    # D: add temporal_asym
    "D_add_temporal_asym": [
        ("brel", "frontal", 4, 8), ("brel", "frontal", 8, 13),
        ("asym", "frontal"), ("asym", "temporal"),
        ("tbr", "frontal"), ("riem", "central", "temporal"),
    ],
    # E: multi-region theta_rel
    "E_multi_theta": [
        ("brel", "frontal", 4, 8), ("brel", "temporal", 4, 8), ("brel", "parietal", 4, 8),
        ("asym", "frontal"), ("tbr", "frontal"), ("riem", "central", "temporal"),
    ],
    # F: minimal 3-dim (frontal_alpha_rel + asym + TBR)
    "F_minimal_3dim": [
        ("brel", "frontal", 8, 13), ("asym", "frontal"), ("tbr", "frontal"),
    ],
    # G: add riem_central_parietal (was 2nd best F-score)
    "G_add_riem_cp": [
        ("brel", "frontal", 4, 8), ("brel", "frontal", 8, 13),
        ("asym", "frontal"), ("tbr", "frontal"),
        ("riem", "central", "temporal"), ("riem", "central", "parietal"),
    ],
    # H: C=0.01 test (via baseline features, tested separately)
    "H_delta_frontal": [
        ("brel", "frontal", 1, 4), ("brel", "frontal", 8, 13),
        ("asym", "frontal"), ("tbr", "frontal"), ("riem", "central", "temporal"),
    ],
}

if __name__ == "__main__":
    X, y, groups, ch_names = load_filtered_data()
    log.info(f"Data: {X.shape[0]} windows, {len(np.unique(groups))} subjects\n")
    results = []
    for name, cfg in CONFIGS.items():
        feats, fnames = make_features(X, ch_names, cfg)
        ba = run_cv(feats, y, groups)
        log.info(f"  {name:25s}  dims={feats.shape[1]:2d}  BA={ba:.3f}")
        results.append((name, feats.shape[1], ba))
    log.info("\n--- Sorted by BA ---")
    for name, d, ba in sorted(results, key=lambda x: -x[2]):
        log.info(f"  {name:25s}  dims={d:2d}  BA={ba:.3f}")
