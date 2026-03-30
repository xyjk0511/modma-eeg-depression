"""Delta feature ablation study.
Compares 3 strategies using exclude_high_fp filtering (Phase 7 baseline BA=0.6913):
  1. baseline: v3_subset only (676-dim combined EO+EC)
  2. delta_only: delta coherence + asymmetry only (44-dim combined)
  3. augmented: v3_subset + delta features (720-dim combined)

Usage:
  python run_delta_ablation.py              # run all experiments
  python run_delta_ablation.py --report-only  # regenerate charts/summary from JSON
"""
import argparse
import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import permutation_test
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from config import PARTICIPANTS_TSV
from extract_delta_features import CACHE_DIR as DELTA_CACHE_DIR
from main_diagnostic import load_discovery_subjects
from run_ablation import load_all, _indices_for_condition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
V3_CATEGORIES = ["abs_bp", "rel_bp", "Hjorth"]
N_SPLITS = 5
RANDOM_STATE = 42

HIGH_FP_EXCLUSIONS = {
    "BURNOUT", "CHRONIC PAIN", "INSOMNIA", "OCD",
    "PARKINSON", "SMC", "TINNITUS",
}

RESULTS_PATH = Path("delta_ablation_results.json")
SUMMARY_PATH = Path("delta_ablation_summary.md")
FIGURE_DIR = Path("figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pipeline(rs=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=rs)),
    ])


def _load_indication_map(subject_ids):
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    df = df[df["participants_ID"].isin(subject_ids)].drop_duplicates(
        subset=["participants_ID"])
    df["indication_clean"] = df["indication"].fillna("UNKNOWN").astype(str).str.strip()
    df.loc[df["indication_clean"].isin(["nan", ""]), "indication_clean"] = "UNKNOWN"
    return dict(zip(df["participants_ID"], df["indication_clean"]))


def _load_delta_features():
    """Load cached delta features for EO and EC, return combined array + subject list."""
    subs_eo = set(load_discovery_subjects("EO")["participants_ID"])
    subs_ec = set(load_discovery_subjects("EC")["participants_ID"])

    def _load_cond(condition, valid_sids):
        cache = DELTA_CACHE_DIR / condition
        feats = {}
        for npz_path in cache.glob("*.npz"):
            sid = npz_path.stem
            if sid not in valid_sids:
                continue
            d = np.load(npz_path)
            # Concatenate coherence (12) + asymmetry (10) = 22
            feat = np.concatenate([d["coherence"], d["asymmetry"]])
            feats[sid] = feat
        return feats

    delta_eo = _load_cond("EO", subs_eo)
    delta_ec = _load_cond("EC", subs_ec)
    common = sorted(set(delta_eo) & set(delta_ec))

    # Combined: EO delta (22) + EC delta (22) = 44
    X_delta = np.array([np.concatenate([delta_eo[s], delta_ec[s]]) for s in common])
    return X_delta, np.array(common)


def _cv_evaluate(X, y_enc, groups, mdd_idx, n_splits=N_SPLITS, desc="CV"):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    fold_bas = []
    all_probas = np.full(len(y_enc), np.nan)
    all_preds = np.full(len(y_enc), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(
            tqdm(cv.split(X, y_enc, groups), total=n_splits, desc=desc)):
        pipe = _make_pipeline()
        pipe.fit(X[train_idx], y_enc[train_idx])
        all_probas[test_idx] = pipe.predict_proba(X[test_idx])[:, mdd_idx]
        all_preds[test_idx] = pipe.predict(X[test_idx])

        y_bin_fold = (y_enc[test_idx] == mdd_idx).astype(int)
        pred_bin_fold = (all_preds[test_idx] == mdd_idx).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin_fold, pred_bin_fold).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fold_bas.append((sen + spe) / 2)

    y_bin = (y_enc == mdd_idx).astype(int)
    pred_bin = (all_preds == mdd_idx).astype(int)
    auc = float(roc_auc_score(y_bin, all_probas))
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ba = (sen + spe) / 2

    return {
        "ba": round(ba, 4),
        "auc": round(auc, 4),
        "sen": round(sen, 4),
        "spe": round(spe, 4),
        "per_fold_ba": [round(b, 4) for b in fold_bas],
    }, all_probas, y_bin


def _permutation_test(fold_ba_exp, fold_ba_base):
    a = np.array(fold_ba_exp)
    b = np.array(fold_ba_base)

    def stat_fn(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = permutation_test(
        (a, b), statistic=stat_fn,
        permutation_type="independent", n_resamples=9999,
        alternative="greater", random_state=RANDOM_STATE)
    return round(float(result.pvalue), 6)

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiments():
    """Run 3-strategy ablation with exclude_high_fp filtering."""
    print("Loading cached features...")
    datasets = load_all()
    X_full, y_raw, groups_full = datasets["combined"]

    # Load delta features
    X_delta_all, delta_sids = _load_delta_features()
    print(f"Delta features: {X_delta_all.shape} for {len(delta_sids)} subjects")

    # Find subjects with BOTH v3_subset and delta features
    full_sid_set = set(groups_full)
    delta_sid_set = set(delta_sids)
    common_sids = sorted(full_sid_set & delta_sid_set)
    print(f"Subjects with both feature sets: {len(common_sids)}")

    # Build aligned arrays
    full_idx_map = {s: i for i, s in enumerate(groups_full)}
    delta_idx_map = {s: i for i, s in enumerate(delta_sids)}

    full_indices = [full_idx_map[s] for s in common_sids]
    delta_indices = [delta_idx_map[s] for s in common_sids]

    X_full_aligned = X_full[full_indices]
    X_delta_aligned = X_delta_all[delta_indices]
    y_raw_aligned = y_raw[full_indices]
    groups = np.array(common_sids)

    # v3_subset features
    v3_idx = _indices_for_condition(V3_CATEGORIES, "combined")
    X_v3 = X_full_aligned[:, v3_idx]
    print(f"v3_subset: {X_v3.shape[1]}-dim, delta: {X_delta_aligned.shape[1]}-dim")

    # Apply exclude_high_fp filtering
    indication_map = _load_indication_map(list(groups))
    indications = np.array([indication_map.get(g, "UNKNOWN") for g in groups])

    def _should_exclude(ind):
        primary = ind.split("/")[0].strip()
        return primary in HIGH_FP_EXCLUSIONS

    mask = np.array([
        not (_should_exclude(indications[i]) and y_raw_aligned[i] == "nonMDD")
        for i in range(len(groups))
    ])

    X_v3_f = X_v3[mask]
    X_delta_f = X_delta_aligned[mask]
    y_f = y_raw_aligned[mask]
    g_f = groups[mask]

    # Handle NaN in delta features: replace with 0 (conservative)
    nan_count = np.isnan(X_delta_f).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values in delta features, filling with 0")
        X_delta_f = np.nan_to_num(X_delta_f, nan=0.0)

    le = LabelEncoder()
    le.fit(["MDD", "nonMDD"])
    y_enc = le.transform(y_f)
    mdd_idx = list(le.classes_).index("MDD")

    n_mdd = int((y_f == "MDD").sum())
    n_nonmdd = int((y_f == "nonMDD").sum())
    print(f"\nexclude_high_fp: n={len(g_f)} (MDD={n_mdd}, nonMDD={n_nonmdd})")

    # Build strategy feature matrices
    X_augmented = np.hstack([X_v3_f, X_delta_f])
    strategies = {
        "baseline": (X_v3_f, f"v3_subset only ({X_v3_f.shape[1]}-dim)"),
        "delta_only": (X_delta_f, f"delta only ({X_delta_f.shape[1]}-dim)"),
        "augmented": (X_augmented, f"v3_subset + delta ({X_augmented.shape[1]}-dim)"),
    }

    results = {"experiments": {}, "roc_data": {}}

    for name, (X_strat, desc) in strategies.items():
        print(f"\n[{name}] {desc}")
        metrics, probas, y_bin = _cv_evaluate(
            X_strat, y_enc, g_f, mdd_idx, desc=name)
        metrics["n_subjects"] = len(g_f)
        metrics["n_mdd"] = n_mdd
        metrics["n_nonmdd"] = n_nonmdd
        metrics["n_features"] = int(X_strat.shape[1])
        metrics["description"] = desc
        results["experiments"][name] = metrics

        fpr, tpr, _ = roc_curve(y_bin, probas)
        results["roc_data"][name] = {
            "fpr": [round(float(v), 6) for v in fpr],
            "tpr": [round(float(v), 6) for v in tpr],
        }
        print(f"  BA={metrics['ba']:.4f}, AUC={metrics['auc']:.4f}, "
              f"SEN={metrics['sen']:.4f}, SPE={metrics['spe']:.4f}")

    # Statistical test: augmented vs baseline
    print("\n[Statistical comparison]")
    base_fold = results["experiments"]["baseline"]["per_fold_ba"]
    results["statistical_tests"] = {}

    for name in ["delta_only", "augmented"]:
        exp = results["experiments"][name]
        fold_ba = exp["per_fold_ba"]
        delta_ba = exp["ba"] - results["experiments"]["baseline"]["ba"]
        p = _permutation_test(fold_ba, base_fold)
        results["statistical_tests"][f"{name}_vs_baseline"] = {
            "delta_ba": round(delta_ba, 4),
            "p_value": p,
            "significant": bool(p < 0.05),
        }
        print(f"  {name} vs baseline: delta-BA={delta_ba:+.4f}, p={p:.6f}")

    # Decision logic
    aug_test = results["statistical_tests"]["augmented_vs_baseline"]
    aug_ba = results["experiments"]["augmented"]["ba"]
    base_ba = results["experiments"]["baseline"]["ba"]

    if aug_ba > base_ba and aug_test["significant"]:
        decision = "adopt"
        rationale = (f"Augmented features significantly improve BA "
                     f"({base_ba:.4f} -> {aug_ba:.4f}, p={aug_test['p_value']})")
    elif aug_ba > base_ba:
        decision = "marginal"
        rationale = (f"Augmented BA improved ({base_ba:.4f} -> {aug_ba:.4f}) "
                     f"but not significant (p={aug_test['p_value']})")
    else:
        decision = "reject"
        rationale = (f"Delta features do not improve BA "
                     f"({base_ba:.4f} -> {aug_ba:.4f})")

    results["decision"] = {
        "recommendation": decision,
        "rationale": rationale,
        "adopt_delta": decision == "adopt",
    }

    results["task"] = "delta_ablation"
    results["timestamp"] = datetime.datetime.now().isoformat()

    _save_results(results)
    return results

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_charts(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)
    experiments = results.get("experiments", {})

    # --- 1. BA comparison bar chart ---
    strategies = ["baseline", "delta_only", "augmented"]
    labels = [f"Baseline\n(v3_subset, {experiments.get('baseline', {}).get('n_features', '?')}d)",
              f"Delta Only\n({experiments.get('delta_only', {}).get('n_features', '?')}d)",
              f"Augmented\n(v3+delta, {experiments.get('augmented', {}).get('n_features', '?')}d)"]
    bas = []
    stds = []
    for s in strategies:
        r = experiments.get(s, {})
        bas.append(r.get("ba", 0))
        fold_ba = r.get("per_fold_ba", [])
        stds.append(float(np.std(fold_ba)) if fold_ba else 0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(strategies))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    bars = ax.bar(x, bas, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.axhline(0.6913, color="red", linestyle="--", linewidth=1.5,
               label="Phase 7 baseline BA=0.6913")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Delta Feature Ablation: BA Comparison (exclude_high_fp)")
    ax.legend(fontsize=9)

    for bar, ba_val, std_val in zip(bars, bas, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std_val + 0.005,
                f"{ba_val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.45, max(bas) + 0.08)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "delta_ablation_ba.png")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'delta_ablation_ba.png'}")

    # --- 2. ROC curves: baseline vs augmented ---
    roc_data = results.get("roc_data", {})
    if roc_data:
        fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)
        colors_roc = {"baseline": "#1f77b4", "augmented": "#ff7f0e"}
        for s in ["baseline", "augmented"]:
            rd = roc_data.get(s, {})
            fpr = rd.get("fpr", [])
            tpr = rd.get("tpr", [])
            auc_val = experiments.get(s, {}).get("auc", 0)
            if fpr and tpr:
                ax2.plot(fpr, tpr, color=colors_roc.get(s, "gray"),
                         label=f"{s} (AUC={auc_val:.4f})", linewidth=2)
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("Delta Ablation: ROC Curves (baseline vs augmented)")
        ax2.legend(fontsize=9, loc="lower right")
        fig2.tight_layout()
        fig2.savefig(FIGURE_DIR / "delta_ablation_roc.png")
        plt.close(fig2)
        print(f"Saved {FIGURE_DIR / 'delta_ablation_roc.png'}")

# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def generate_summary(results):
    experiments = results.get("experiments", {})
    stats = results.get("statistical_tests", {})
    decision = results.get("decision", {})
    baseline = experiments.get("baseline", {})

    lines = []
    lines.append("# Delta Feature Ablation Summary\n\n")

    lines.append("## 1. Experiment Setup\n\n")
    lines.append("- **Pipeline:** StandardScaler + SVC(kernel=rbf, "
                 "class_weight=balanced, random_state=42)\n")
    lines.append("- **Subject filter:** exclude_high_fp (Phase 7)\n")
    lines.append(f"- **CV:** {N_SPLITS}-fold StratifiedGroupKFold\n")
    lines.append("- **Statistical test:** scipy permutation_test, "
                 "permutation_type=independent, n_resamples=9999\n\n")

    lines.append("## 2. Feature Sets\n\n")
    lines.append("| Strategy | Features | Dims |\n")
    lines.append("|----------|----------|------|\n")
    for s in ["baseline", "delta_only", "augmented"]:
        r = experiments.get(s, {})
        lines.append(f"| {s} | {r.get('description', '')} | "
                     f"{r.get('n_features', '?')} |\n")
    lines.append("\n")

    lines.append("## 3. Results Comparison\n\n")
    lines.append("| Strategy | BA | AUC | SEN | SPE | delta-BA | p-value |\n")
    lines.append("|----------|-----|-----|-----|-----|----------|--------|\n")

    base_ba = baseline.get("ba", 0)
    lines.append(f"| baseline | {base_ba:.4f} | "
                 f"{baseline.get('auc', 0):.4f} | "
                 f"{baseline.get('sen', 0):.4f} | "
                 f"{baseline.get('spe', 0):.4f} | - | - |\n")

    for s in ["delta_only", "augmented"]:
        r = experiments.get(s, {})
        ba = r.get("ba", 0)
        delta = ba - base_ba
        st = stats.get(f"{s}_vs_baseline", {})
        p = st.get("p_value", "N/A")
        p_str = f"{p:.6f}" if isinstance(p, float) else str(p)
        lines.append(f"| {s} | {ba:.4f} | {r.get('auc', 0):.4f} | "
                     f"{r.get('sen', 0):.4f} | {r.get('spe', 0):.4f} | "
                     f"{delta:+.4f} | {p_str} |\n")
    lines.append("\n")

    lines.append("## 4. Per-fold BA\n\n")
    lines.append("| Fold | baseline | delta_only | augmented |\n")
    lines.append("|------|----------|------------|----------|\n")
    for i in range(N_SPLITS):
        b = experiments.get("baseline", {}).get("per_fold_ba", [0]*N_SPLITS)
        d = experiments.get("delta_only", {}).get("per_fold_ba", [0]*N_SPLITS)
        a = experiments.get("augmented", {}).get("per_fold_ba", [0]*N_SPLITS)
        lines.append(f"| {i+1} | {b[i]:.4f} | {d[i]:.4f} | {a[i]:.4f} |\n")
    lines.append("\n")

    lines.append("## 5. Decision\n\n")
    lines.append(f"- **Recommendation:** {decision.get('recommendation', 'N/A')}\n")
    lines.append(f"- **Rationale:** {decision.get('rationale', 'N/A')}\n")
    lines.append(f"- **Adopt delta features:** "
                 f"{decision.get('adopt_delta', 'N/A')}\n\n")

    lines.append("## 6. Notes\n\n")
    lines.append("- Delta coherence computed via scipy.signal.coherence "
                 "(no mne-connectivity dependency)\n")
    lines.append("- NaN delta features (from failed coherence computation) "
                 "filled with 0\n")
    lines.append("- Phase 7 baseline (exclude_high_fp, v3_subset): "
                 "BA=0.6913\n")

    with open(SUMMARY_PATH, "w", newline="\n", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved {SUMMARY_PATH}")

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"experiments": {}, "statistical_tests": {}, "roc_data": {}}


def _save_results(results):
    with open(RESULTS_PATH, "w", newline="\n", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Delta feature ablation study.")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate charts/summary from existing JSON")
    args = parser.parse_args()

    if args.report_only:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} not found. Run full analysis first.")
            return
        results = _load_results()
        generate_charts(results)
        generate_summary(results)
        return

    results = run_experiments()
    generate_charts(results)
    generate_summary(results)

    # Final summary
    exp = results["experiments"]
    dec = results["decision"]
    print(f"\n{'=' * 60}")
    print("Delta Feature Ablation Complete")
    print(f"  baseline:   BA={exp['baseline']['ba']:.4f} ({exp['baseline']['n_features']}d)")
    print(f"  delta_only: BA={exp['delta_only']['ba']:.4f} ({exp['delta_only']['n_features']}d)")
    print(f"  augmented:  BA={exp['augmented']['ba']:.4f} ({exp['augmented']['n_features']}d)")
    print(f"  Decision: {dec['recommendation']} — {dec['rationale']}")


if __name__ == "__main__":
    main()
