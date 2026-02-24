import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT
"""Reproduce 3 literature methods on MODMA with strict subject-level CV.

Methods:
  1. Band-specific Riemannian (theta/alpha/beta covariance tangent space)
  2. Nonlinear features (sample entropy, permutation entropy, Hjorth)
  3. PLV connectivity matrix (all region-pair PLVs across bands)

All use StratifiedGroupKFold + subject-level aggregation.
"""
import argparse, json, logging, os, time, warnings
import numpy as np
from scipy.signal import butter, sosfilt, hilbert as sig_hilbert
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import antropy

from modma_mdd_real_experiment import (
    load_participants, load_windows, build_region_indices,
    compute_subject_balanced_sample_weights, build_quality_mask,
)

logger = logging.getLogger(__name__)

BIDS_ROOT = str(ROOT / "data/modma_bids/EEG_LZU_2015_2_resting state")
SFREQ = 125.0
BANDS = [(4, 8), (8, 13), (13, 30)]
REGIONS = ["frontal", "central", "temporal", "parietal"]


# ── Helpers ──────────────────────────────────────────────────────────────

def bandpass(X, fmin, fmax, sfreq):
    sos = butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")
    return sosfilt(sos, X, axis=2)


def subject_aggregate(groups, y, probs):
    yt_s, yp_s = {}, {}
    for i, g in enumerate(groups):
        if g not in yt_s:
            yt_s[g] = y[i]
            yp_s[g] = []
        yp_s[g].append(probs[i])
    subjs = list(yt_s.keys())
    return (np.array([yt_s[s] for s in subjs]),
            np.array([np.mean(yp_s[s]) for s in subjs]))


def eval_ba_auc(yt, yp):
    ba = balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
    auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5
    return round(ba, 4), round(auc, 4)


# ── Method 1: Band-specific Riemannian ──────────────────────────────────

def extract_riemann(X_tr, X_te, est="lwf"):
    cov = Covariances(estimator=est)
    ts = TangentSpace()
    F_tr = ts.fit_transform(cov.fit_transform(X_tr))
    F_te = ts.transform(cov.transform(X_te))
    return F_tr, F_te


def _select_region_channels(X, ch_names, n_per_region=5):
    """Select top n channels per region for manageable Riemannian."""
    if ch_names is None:
        return X
    ridx = build_region_indices(ch_names)
    selected = []
    for r in REGIONS:
        idx = ridx.get(r, [])
        if idx:
            selected.extend(idx[:n_per_region])
    if not selected:
        return X
    return X[:, selected, :]


def method_riemannian(X, y, groups, n_splits, seed, ch_names=None):
    """Band-specific Riemannian on ~20 selected channels + Hjorth."""
    X_sel = _select_region_channels(X, ch_names, n_per_region=5)
    cv = StratifiedGroupKFold(n_splits=n_splits)
    all_g, all_y, all_p = [], [], []

    for tr_ix, te_ix in cv.split(X_sel, y, groups=groups):
        X_tr, X_te = X_sel[tr_ix], X_sel[te_ix]
        y_tr, g_te = y[tr_ix], groups[te_ix]
        sw = compute_subject_balanced_sample_weights(groups[tr_ix])

        parts_tr, parts_te = [], []
        for fmin, fmax in BANDS:
            Xb_tr = bandpass(X_tr, fmin, fmax, SFREQ)
            Xb_te = bandpass(X_te, fmin, fmax, SFREQ)
            Ft, Fe = extract_riemann(Xb_tr, Xb_te, "lwf")
            parts_tr.append(Ft); parts_te.append(Fe)

        parts_tr.append(_compute_hjorth(X[tr_ix], ch_names))
        parts_te.append(_compute_hjorth(X[te_ix], ch_names))

        Ftr = np.column_stack(parts_tr)
        Fte = np.column_stack(parts_te)

        pca_n = min(30, Ftr.shape[1])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_n, random_state=seed)),
            ("clf", LogisticRegression(C=10, max_iter=2000, random_state=seed)),
        ])
        pipe.fit(Ftr, y_tr, clf__sample_weight=sw)
        probs = pipe.predict_proba(Fte)[:, 1]
        all_g.extend(g_te); all_y.extend(y[te_ix]); all_p.extend(probs)

    yt, yp = subject_aggregate(np.array(all_g), np.array(all_y), np.array(all_p))
    return eval_ba_auc(yt, yp)


def _get_region_slices(ch_names):
    """Get region channel indices using proper 10-20/EGI mapping."""
    if ch_names is not None:
        ridx = build_region_indices(ch_names)
        return [ridx[r] for r in REGIONS if ridx[r]]
    n_ch = 128
    q = n_ch // 4
    return [list(range(0, q)), list(range(q, 2*q)),
            list(range(2*q, 3*q)), list(range(3*q, n_ch))]


def _compute_hjorth(X, ch_names=None):
    """Hjorth params (activity, mobility, complexity) per region."""
    feats = []
    for idx in _get_region_slices(ch_names):
        sig = np.mean(X[:, idx, :], axis=1)
        var0 = np.var(sig, axis=1, keepdims=True)
        d1 = np.diff(sig, axis=1)
        var1 = np.var(d1, axis=1, keepdims=True)
        d2 = np.diff(d1, axis=1)
        var2 = np.var(d2, axis=1, keepdims=True)
        mob = np.sqrt(var1 / (var0 + 1e-10))
        comp = np.sqrt(var2 / (var1 + 1e-10)) / (mob + 1e-10)
        feats.extend([np.log(var0 + 1e-10), mob, comp])
    return np.column_stack(feats)


# ── Method 2: Nonlinear features ────────────────────────────────────────

def method_nonlinear(X, y, groups, n_splits, seed, ch_names=None):
    """Sample entropy + permutation entropy + Hjorth per band per region."""
    feats = _extract_nonlinear_features(X, ch_names)
    return _run_cv_on_features(feats, y, groups, n_splits, seed, C=1.0)


def _extract_nonlinear_features(X, ch_names=None):
    n_win, n_ch, n_t = X.shape
    region_indices = _get_region_slices(ch_names)
    all_feats = []

    for wi in range(n_win):
        row = []
        for idx in region_indices:
            sig = np.mean(X[wi, idx, :], axis=0)
            for fmin, fmax in BANDS:
                filtered = _bp_1d(sig, fmin, fmax, SFREQ)
                try:
                    se = antropy.sample_entropy(filtered)
                except Exception:
                    se = 0.0
                try:
                    pe = antropy.perm_entropy(filtered, normalize=True)
                except Exception:
                    pe = 0.0
                var0 = np.var(filtered)
                d1 = np.diff(filtered)
                var1 = np.var(d1)
                mob = np.sqrt(var1 / (var0 + 1e-10))
                row.extend([se, pe, mob])
        all_feats.append(row)

    return np.array(all_feats)


def _bp_1d(sig, fmin, fmax, sfreq):
    sos = butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")
    return sosfilt(sos, sig)


# ── Method 3: PLV connectivity matrix ───────────────────────────────────

def method_plv_matrix(X, y, groups, n_splits, seed, ch_names=None):
    """Full PLV matrix between all region pairs across bands."""
    feats = _extract_plv_features(X, ch_names)
    return _run_cv_on_features(feats, y, groups, n_splits, seed, C=0.1)


def _extract_plv_features(X, ch_names=None):
    region_indices = _get_region_slices(ch_names)
    pairs = [(i, j) for i in range(len(region_indices)) for j in range(i+1, len(region_indices))]

    all_feats = []
    for fmin, fmax in BANDS:
        Xb = bandpass(X, fmin, fmax, SFREQ)
        for ri, rj in pairs:
            sig_a = np.mean(Xb[:, region_indices[ri], :], axis=1)
            sig_b = np.mean(Xb[:, region_indices[rj], :], axis=1)
            pa = np.angle(sig_hilbert(sig_a, axis=1))
            pb = np.angle(sig_hilbert(sig_b, axis=1))
            plv = np.abs(np.mean(np.exp(1j * (pa - pb)), axis=1))
            all_feats.append(plv[:, np.newaxis])

    return np.column_stack(all_feats)


# ── Shared CV runner ────────────────────────────────────────────────────

def _run_cv_on_features(feats, y, groups, n_splits, seed, C=1.0):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    all_g, all_y, all_p = [], [], []

    for tr_ix, te_ix in cv.split(feats, y, groups=groups):
        sw = compute_subject_balanced_sample_weights(groups[tr_ix])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed)),
        ])
        pipe.fit(feats[tr_ix], y[tr_ix], clf__sample_weight=sw)
        probs = pipe.predict_proba(feats[te_ix])[:, 1]
        all_g.extend(groups[te_ix])
        all_y.extend(y[te_ix])
        all_p.extend(probs)

    yt, yp = subject_aggregate(np.array(all_g), np.array(all_y), np.array(all_p))
    return eval_ba_auc(yt, yp)


# ── Permutation test ────────────────────────────────────────────────────

def permutation_test(method_fn, X, y, groups, n_splits, seed, observed_ba, n_perm=200, ch_names=None):
    rng = np.random.RandomState(seed)
    unique_g, first_idx = np.unique(groups, return_index=True)
    g_labels = y[first_idx]
    count = 0
    for i in range(n_perm):
        perm_labels = rng.permutation(g_labels)
        if len(np.unique(perm_labels)) < 2:
            continue
        label_map = dict(zip(unique_g, perm_labels))
        y_perm = np.array([label_map[g] for g in groups])
        try:
            ba, _ = method_fn(X, y_perm, groups, n_splits, seed, ch_names=ch_names)
            if ba >= observed_ba:
                count += 1
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            logger.info(f"  perm {i+1}/{n_perm}")
    return round((count + 1) / (n_perm + 1), 4)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", default=BIDS_ROOT)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/results_modma_reproduce"))
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Load data
    logger.info("Loading MODMA resting-state EEG...")
    df = load_participants(os.path.join(args.bids_root, "participants.tsv"))
    X, y, groups, no_edf, ch_names, interp = load_windows(
        df, bids_root=args.bids_root, window_sec=10, resample_sfreq=SFREQ,
        crop_duration=60.0, highpass_freq=1.0, bad_amp_uv=200.0,
        cache_dir=os.path.join(args.output_dir, ".cache"),
    )

    # QC filter
    n_ch = X.shape[1]
    keep = build_quality_mask(X, bad_amp_uv=200.0, max_bad_channels=max(1, int(n_ch * 0.12)))
    unique_g, counts = np.unique(groups[keep], return_counts=True)
    valid_g = unique_g[counts >= 3]
    mask = keep & np.isin(groups, valid_g)
    X, y, groups = X[mask], y[mask], groups[mask]

    unique_g, first_idx = np.unique(groups, return_index=True)
    g_labels = y[first_idx]
    n_mdd = np.sum(g_labels == 1)
    n_hc = np.sum(g_labels == 0)
    n_splits = min(5, min(n_mdd, n_hc))
    if n_splits < 2:
        n_splits = 2
    logger.info(f"Subjects: {n_mdd} MDD, {n_hc} HC | Windows: {len(y)} | Splits: {n_splits}")

    # Run 3 methods
    methods = {
        "riemannian_band_hjorth_pca30": method_riemannian,
        "nonlinear_entropy_hjorth": method_nonlinear,
        "plv_connectivity_matrix": method_plv_matrix,
    }

    results = {}
    for name, fn in methods.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Method: {name}")
        logger.info(f"{'='*60}")
        try:
            ba, auc = fn(X, y, groups, n_splits, args.seed, ch_names=ch_names)
            logger.info(f"  BA={ba:.4f}  AUC={auc:.4f}")

            # Permutation test
            logger.info(f"  Permutation test ({args.n_permutations} perms)...")
            p = permutation_test(fn, X, y, groups, n_splits, args.seed, ba,
                                args.n_permutations, ch_names=ch_names)
            logger.info(f"  p-value={p:.4f}")

            results[name] = {"ba": ba, "auc": auc, "p_value": p}
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results[name] = {"error": str(e)}

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for name, r in results.items():
        if "error" in r:
            logger.info(f"  {name}: FAILED - {r['error']}")
        else:
            sig = "***" if r["p_value"] < 0.05 else ""
            logger.info(f"  {name}: BA={r['ba']:.4f} AUC={r['auc']:.4f} p={r['p_value']:.4f} {sig}")

    output = {
        "dataset": "MODMA_resting_state",
        "n_mdd": int(n_mdd), "n_hc": int(n_hc), "n_windows": int(len(y)),
        "n_splits": n_splits,
        "methods": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    path = os.path.join(args.output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
