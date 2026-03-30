"""Age covariate integration experiments.
Tests whether adding age as a feature or training age-stratified models
improves classification performance over the pooled baseline (BA=0.6318).

Experiments:
  1. Pooled baseline (v3_subset, 676-dim)
  2. Age-as-feature (v3_subset + age, 677-dim)
  3. Age-stratified models (young/middle/old separate SVMs)
  4. Age-interaction features (v3_subset + age + age*top5_delta, 682-dim)

Usage:
  python run_age_covariate.py            # run all experiments
  python run_age_covariate.py --report-only  # regenerate charts/summary from JSON
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

from run_ablation import load_all, _indices_for_condition
from config import PARTICIPANTS_TSV

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
V3_CATEGORIES = ["abs_bp", "rel_bp", "Hjorth"]
N_SPLITS = 5
RANDOM_STATE = 42

# Age tertile boundaries from error_analysis.json
AGE_TERTILE_LOW = 26.7
AGE_TERTILE_HIGH = 49.1

RESULTS_PATH = Path("age_covariate_results.json")
SUMMARY_PATH = Path("age_covariate_summary.md")
FIGURE_DIR = Path("figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pipeline(rs=RANDOM_STATE):
    """StandardScaler + SVC(rbf, balanced) pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=rs)),
    ])

def _load_age_data(subject_ids):
    """Load age from participants.tsv, return dict {sid: age_float}.
    Uses comma-decimal format parsing (same as run_error_analysis.py)."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    df = df[df["participants_ID"].isin(subject_ids)].drop_duplicates(
        subset=["participants_ID"])
    df["age_float"] = df["age"].astype(str).str.replace(",", ".").astype(float)
    age_map = dict(zip(df["participants_ID"], df["age_float"]))
    return age_map


def _cv_evaluate(X, y_enc, groups, mdd_idx, n_splits=N_SPLITS, desc="CV"):
    """Run StratifiedGroupKFold CV, return per-fold BA and overall metrics."""
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


def _paired_permutation_test(fold_ba_exp, fold_ba_base):
    """Exact permutation test on paired fold BA differences (2^5=32 perms)."""
    diffs = np.array(fold_ba_exp) - np.array(fold_ba_base)

    def stat_fn(x, axis):
        return np.mean(x, axis=axis)

    result = permutation_test(
        (diffs,), statistic=stat_fn,
        permutation_type="samples", n_resamples=9999,
        alternative="greater", random_state=RANDOM_STATE)
    return round(float(result.pvalue), 6)


# ---------------------------------------------------------------------------
# Experiment 1: Pooled baseline
# ---------------------------------------------------------------------------
def run_pooled_baseline(X_v3, y_enc, groups, mdd_idx):
    """Re-run pooled baseline for consistent comparison."""
    print("\n[Experiment 1] Pooled baseline (676-dim)")
    metrics, probas, y_bin = _cv_evaluate(
        X_v3, y_enc, groups, mdd_idx, desc="Baseline")
    print(f"  BA={metrics['ba']:.4f}, AUC={metrics['auc']:.4f}")
    return metrics, probas, y_bin


# ---------------------------------------------------------------------------
# Experiment 2: Age-as-feature
# ---------------------------------------------------------------------------
def run_age_as_feature(X_v3, y_enc, groups, mdd_idx, age_map):
    """Append standardized age as 677th feature."""
    # Filter to subjects with valid age
    mask = np.array([not np.isnan(age_map.get(g, np.nan)) for g in groups])
    X_sub = X_v3[mask]
    y_sub = y_enc[mask]
    g_sub = groups[mask]
    ages = np.array([age_map[g] for g in g_sub]).reshape(-1, 1)

    X_aug = np.hstack([X_sub, ages])
    print(f"\n[Experiment 2] Age-as-feature ({X_aug.shape[1]}-dim, "
          f"n={len(g_sub)})")
    metrics, probas, y_bin = _cv_evaluate(
        X_aug, y_sub, g_sub, mdd_idx, desc="Age-as-feature")
    metrics["n_subjects"] = int(len(g_sub))
    print(f"  BA={metrics['ba']:.4f}, AUC={metrics['auc']:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Experiment 3: Age-stratified models
# ---------------------------------------------------------------------------
def run_age_stratified(X_v3, y_enc, groups, mdd_idx, age_map):
    """Train separate SVM per age bin, route test subjects by age."""
    mask = np.array([not np.isnan(age_map.get(g, np.nan)) for g in groups])
    X_sub = X_v3[mask]
    y_sub = y_enc[mask]
    g_sub = groups[mask]
    ages = np.array([age_map[g] for g in g_sub])

    def _age_bin(a):
        if a <= AGE_TERTILE_LOW:
            return "young"
        elif a <= AGE_TERTILE_HIGH:
            return "middle"
        else:
            return "old"

    bins = np.array([_age_bin(a) for a in ages])
    print(f"\n[Experiment 3] Age-stratified (n={len(g_sub)})")
    for b in ["young", "middle", "old"]:
        n = np.sum(bins == b)
        n_mdd = np.sum((bins == b) & (y_sub == mdd_idx))
        print(f"  {b}: n={n}, MDD={n_mdd}")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    all_preds = np.full(len(y_sub), -1, dtype=int)
    all_probas = np.full(len(y_sub), np.nan)
    fold_bas = []

    for fold_i, (train_idx, test_idx) in enumerate(
            tqdm(cv.split(X_sub, y_sub, g_sub), total=N_SPLITS,
                 desc="Age-stratified")):
        train_bins = bins[train_idx]
        test_bins = bins[test_idx]

        # Train pooled fallback model
        pooled_pipe = _make_pipeline()
        pooled_pipe.fit(X_sub[train_idx], y_sub[train_idx])

        for b in ["young", "middle", "old"]:
            train_mask_b = train_bins == b
            test_mask_b = test_bins == b
            if not np.any(test_mask_b):
                continue

            train_b = train_idx[train_mask_b]
            test_b = test_idx[test_mask_b]

            # Check minimum class counts
            y_train_b = y_sub[train_b]
            n_mdd_train = np.sum(y_train_b == mdd_idx)
            n_non_train = np.sum(y_train_b != mdd_idx)

            if n_mdd_train < 5 or n_non_train < 5:
                # Fallback to pooled model
                all_probas[test_b] = pooled_pipe.predict_proba(
                    X_sub[test_b])[:, mdd_idx]
                all_preds[test_b] = pooled_pipe.predict(X_sub[test_b])
            else:
                pipe_b = _make_pipeline()
                pipe_b.fit(X_sub[train_b], y_sub[train_b])
                all_probas[test_b] = pipe_b.predict_proba(
                    X_sub[test_b])[:, mdd_idx]
                all_preds[test_b] = pipe_b.predict(X_sub[test_b])

        # Per-fold BA
        y_bin_fold = (y_sub[test_idx] == mdd_idx).astype(int)
        pred_bin_fold = (all_preds[test_idx] == mdd_idx).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin_fold, pred_bin_fold).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fold_bas.append((sen + spe) / 2)

    # Overall metrics
    y_bin = (y_sub == mdd_idx).astype(int)
    pred_bin = (all_preds == mdd_idx).astype(int)
    auc = float(roc_auc_score(y_bin, all_probas))
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ba = (sen + spe) / 2

    # Per-bin metrics
    per_bin = {}
    for b in ["young", "middle", "old"]:
        b_mask = bins == b
        if not np.any(b_mask):
            continue
        y_b = y_bin[b_mask]
        p_b = pred_bin[b_mask]
        pr_b = all_probas[b_mask]
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_b, p_b).ravel()
        sen_b = float(tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0.0
        spe_b = float(tn_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0
        auc_b = float(roc_auc_score(y_b, pr_b)) if len(np.unique(y_b)) > 1 else 0.0
        per_bin[b] = {
            "n": int(np.sum(b_mask)),
            "ba": round((sen_b + spe_b) / 2, 4),
            "auc": round(auc_b, 4),
            "sen": round(sen_b, 4),
            "spe": round(spe_b, 4),
        }

    metrics = {
        "overall_ba": round(ba, 4),
        "overall_auc": round(auc, 4),
        "overall_sen": round(sen, 4),
        "overall_spe": round(spe, 4),
        "per_fold_ba": [round(b, 4) for b in fold_bas],
        "per_bin": per_bin,
        "n_subjects": int(len(g_sub)),
    }
    print(f"  Overall BA={ba:.4f}, AUC={auc:.4f}")
    for b, m in per_bin.items():
        print(f"  {b}: BA={m['ba']:.4f}, AUC={m['auc']:.4f}, n={m['n']}")
    return metrics, all_probas, y_bin


# ---------------------------------------------------------------------------
# Experiment 4: Age-interaction features
# ---------------------------------------------------------------------------
def run_age_interaction(X_v3, y_enc, groups, mdd_idx, age_map):
    """Create age * top-5 abs_bp_delta interaction features."""
    mask = np.array([not np.isnan(age_map.get(g, np.nan)) for g in groups])
    X_sub = X_v3[mask]
    y_sub = y_enc[mask]
    g_sub = groups[mask]
    ages = np.array([age_map[g] for g in g_sub]).reshape(-1, 1)

    # Top-5 delta-band abs_bp channels = indices 0-4 in v3_subset
    # (abs_bp_delta is the first band, 26 channels, we take first 5)
    top5_delta = X_sub[:, :5]
    interactions = ages * top5_delta  # (n, 5)

    X_aug = np.hstack([X_sub, ages, interactions])
    print(f"\n[Experiment 4] Age-interaction ({X_aug.shape[1]}-dim, "
          f"n={len(g_sub)})")
    metrics, probas, y_bin = _cv_evaluate(
        X_aug, y_sub, g_sub, mdd_idx, desc="Age-interaction")
    metrics["n_subjects"] = int(len(g_sub))
    print(f"  BA={metrics['ba']:.4f}, AUC={metrics['auc']:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_charts(results):
    """Generate comparison bar chart and ROC curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)

    # --- 1. BA comparison bar chart ---
    strategies = ["baseline", "age_as_feature", "age_stratified",
                  "age_interaction"]
    labels = ["Pooled\nBaseline", "Age-as-\nFeature",
              "Age-\nStratified", "Age-\nInteraction"]
    bas = []
    stds = []
    for s in strategies:
        r = results.get(s, {})
        if s == "age_stratified":
            ba = r.get("overall_ba", 0)
            fold_ba = r.get("per_fold_ba", [])
        else:
            ba = r.get("ba", 0)
            fold_ba = r.get("per_fold_ba", [])
        bas.append(ba)
        stds.append(float(np.std(fold_ba)) if fold_ba else 0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(strategies))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(x, bas, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.axhline(0.6318, color="gray", linestyle="--", linewidth=1.5,
               label="Previous baseline BA=0.6318")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Age Covariate Integration: BA Comparison")
    ax.legend(fontsize=9)

    # Add value labels
    for bar, ba_val, std_val in zip(bars, bas, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std_val + 0.005,
                f"{ba_val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.5, max(bas) + 0.08)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "age_covariate_ba_comparison.png")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'age_covariate_ba_comparison.png'}")

    # --- 2. Age-stratified ROC (if data available) ---
    # This requires stored probas; generate from results if available
    strat = results.get("age_stratified", {})
    per_bin = strat.get("per_bin", {})
    if per_bin:
        fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)
        bin_colors = {"young": "#2ca02c", "middle": "#ff7f0e", "old": "#d62728"}
        for b, m in per_bin.items():
            ax2.bar(b, m["ba"], color=bin_colors.get(b, "gray"), alpha=0.8)
            ax2.text(b, m["ba"] + 0.01,
                     f"BA={m['ba']:.3f}\nn={m['n']}", ha="center", fontsize=9)
        ax2.axhline(results.get("baseline", {}).get("ba", 0.6318),
                     color="gray", linestyle="--",
                     label="Pooled baseline")
        ax2.set_ylabel("Balanced Accuracy")
        ax2.set_title("Age-Stratified Model: Per-Bin Performance")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0.4, max(m["ba"] for m in per_bin.values()) + 0.1)
        fig2.tight_layout()
        fig2.savefig(FIGURE_DIR / "age_stratified_roc.png")
        plt.close(fig2)
        print(f"Saved {FIGURE_DIR / 'age_stratified_roc.png'}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def generate_summary(results):
    """Generate human-readable markdown summary."""
    baseline = results["baseline"]
    lines = []
    lines.append("# Age Covariate Integration Summary\n\n")

    # Section 1: Setup
    lines.append("## 1. Experiment Setup\n\n")
    lines.append("- **Pipeline:** StandardScaler + SVC(kernel=rbf, "
                 "class_weight=balanced, random_state=42)\n")
    lines.append("- **Base features:** v3_subset (abs_bp + rel_bp + Hjorth, "
                 "676-dim, combined EO+EC)\n")
    lines.append(f"- **CV:** {N_SPLITS}-fold StratifiedGroupKFold\n")
    lines.append(f"- **Age tertiles:** young <= {AGE_TERTILE_LOW}, "
                 f"middle <= {AGE_TERTILE_HIGH}, old > {AGE_TERTILE_HIGH}\n")
    n_age = results.get("age_as_feature", {}).get("n_subjects", "?")
    lines.append(f"- **Subjects with valid age:** {n_age}\n\n")

    # Section 2: Results table
    lines.append("## 2. Results Comparison\n\n")
    lines.append("| Strategy | BA | AUC | SEN | SPE | delta-BA | p-value |\n")
    lines.append("|----------|-----|-----|-----|-----|----------|--------|\n")

    base_ba = baseline["ba"]
    lines.append(f"| Pooled Baseline | {base_ba:.4f} | "
                 f"{baseline['auc']:.4f} | {baseline['sen']:.4f} | "
                 f"{baseline['spe']:.4f} | - | - |\n")

    for key, label in [("age_as_feature", "Age-as-Feature"),
                       ("age_stratified", "Age-Stratified"),
                       ("age_interaction", "Age-Interaction")]:
        r = results.get(key, {})
        ba = r.get("overall_ba", r.get("ba", 0))
        auc = r.get("overall_auc", r.get("auc", 0))
        sen = r.get("overall_sen", r.get("sen", 0))
        spe = r.get("overall_spe", r.get("spe", 0))
        delta = ba - base_ba
        p = r.get("vs_baseline_p", "N/A")
        p_str = f"{p:.6f}" if isinstance(p, float) else str(p)
        lines.append(f"| {label} | {ba:.4f} | {auc:.4f} | {sen:.4f} | "
                     f"{spe:.4f} | {delta:+.4f} | {p_str} |\n")
    lines.append("\n")

    # Section 3: Age-stratified breakdown
    strat = results.get("age_stratified", {})
    per_bin = strat.get("per_bin", {})
    if per_bin:
        lines.append("## 3. Age-Stratified Breakdown\n\n")
        lines.append("| Age Bin | n | BA | AUC | SEN | SPE |\n")
        lines.append("|---------|---|-----|-----|-----|-----|\n")
        for b in ["young", "middle", "old"]:
            m = per_bin.get(b, {})
            lines.append(f"| {b} | {m.get('n', 0)} | {m.get('ba', 0):.4f} | "
                         f"{m.get('auc', 0):.4f} | {m.get('sen', 0):.4f} | "
                         f"{m.get('spe', 0):.4f} |\n")
        lines.append("\n")

    # Section 4: Best strategy
    best = results.get("best_strategy", {})
    lines.append("## 4. Best Strategy\n\n")
    lines.append(f"- **Name:** {best.get('name', 'N/A')}\n")
    lines.append(f"- **BA:** {best.get('ba', 0):.4f}\n")
    lines.append(f"- **Delta-BA over baseline:** "
                 f"{best.get('delta_ba', 0):+.4f}\n")
    lines.append(f"- **Statistically significant (p<0.05):** "
                 f"{best.get('significant', False)}\n\n")

    # Section 5: Conclusion
    lines.append("## 5. Conclusion\n\n")
    if best.get("significant", False):
        lines.append(f"Age integration via **{best.get('name', '')}** provides "
                     f"a statistically significant improvement of "
                     f"{best.get('delta_ba', 0):+.4f} BA over the pooled "
                     f"baseline (BA={base_ba:.4f}).\n")
    else:
        lines.append(f"No age integration strategy provides a statistically "
                     f"significant improvement over the pooled baseline "
                     f"(BA={base_ba:.4f}). The best strategy "
                     f"({best.get('name', 'N/A')}) achieved "
                     f"BA={best.get('ba', 0):.4f} "
                     f"(delta={best.get('delta_ba', 0):+.4f}, "
                     f"p={best.get('p_value', 'N/A')}).\n")

    with open(SUMMARY_PATH, "w", newline="\n", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _load_results():
    """Load existing results if available."""
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_results(results):
    """Save results to JSON with LF line endings."""
    with open(RESULTS_PATH, "w", newline="\n", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Age covariate integration experiments.")
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

    # --- Full experiment run ---
    print("Loading cached features...")
    datasets = load_all()
    X_full, y_raw, groups = datasets["combined"]

    le = LabelEncoder()
    le.fit(["MDD", "nonMDD"])
    y_enc = le.transform(y_raw)
    mdd_idx = list(le.classes_).index("MDD")

    # v3_subset features
    v3_idx = _indices_for_condition(V3_CATEGORIES, "combined")
    X_v3 = X_full[:, v3_idx]
    print(f"  combined: X={X_full.shape}, v3_subset: {X_v3.shape[1]}-dim")

    # Load age data
    age_map = _load_age_data(list(groups))
    n_valid_age = sum(1 for g in groups if not np.isnan(age_map.get(g, np.nan)))
    n_nan_age = len(groups) - n_valid_age
    print(f"  Age: {n_valid_age} valid, {n_nan_age} NaN")

    # Check for existing results (checkpoint/resume)
    results = _load_results()

    # Experiment 1: Pooled baseline
    if "baseline" not in results:
        metrics, base_probas, base_y_bin = run_pooled_baseline(
            X_v3, y_enc, groups, mdd_idx)
        results["baseline"] = metrics
        _save_results(results)
    else:
        print("\n[Experiment 1] Pooled baseline — already done, skipping")

    # Experiment 2: Age-as-feature
    if "age_as_feature" not in results:
        metrics = run_age_as_feature(X_v3, y_enc, groups, mdd_idx, age_map)
        results["age_as_feature"] = metrics
        _save_results(results)
    else:
        print("\n[Experiment 2] Age-as-feature — already done, skipping")

    # Experiment 3: Age-stratified
    if "age_stratified" not in results:
        metrics, strat_probas, strat_y_bin = run_age_stratified(
            X_v3, y_enc, groups, mdd_idx, age_map)
        results["age_stratified"] = metrics
        _save_results(results)
    else:
        print("\n[Experiment 3] Age-stratified — already done, skipping")

    # Experiment 4: Age-interaction
    if "age_interaction" not in results:
        metrics = run_age_interaction(X_v3, y_enc, groups, mdd_idx, age_map)
        results["age_interaction"] = metrics
        _save_results(results)
    else:
        print("\n[Experiment 4] Age-interaction — already done, skipping")

    # --- Statistical comparison ---
    print("\n[Statistical comparison]")
    base_fold_ba = results["baseline"]["per_fold_ba"]

    for key in ["age_as_feature", "age_stratified", "age_interaction"]:
        r = results[key]
        fold_ba = r.get("per_fold_ba", [])
        if len(fold_ba) == len(base_fold_ba):
            p = _paired_permutation_test(fold_ba, base_fold_ba)
            r["vs_baseline_p"] = p
            ba = r.get("overall_ba", r.get("ba", 0))
            print(f"  {key}: delta-BA={ba - results['baseline']['ba']:+.4f}, "
                  f"p={p:.6f}")
        else:
            r["vs_baseline_p"] = None
            print(f"  {key}: fold count mismatch, skipping test")

    # --- Identify best strategy ---
    candidates = []
    for key in ["baseline", "age_as_feature", "age_stratified",
                "age_interaction"]:
        r = results[key]
        ba = r.get("overall_ba", r.get("ba", 0))
        p = r.get("vs_baseline_p", 1.0)
        candidates.append((key, ba, p))

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_key, best_ba, best_p = candidates[0]
    base_ba = results["baseline"]["ba"]

    results["best_strategy"] = {
        "name": best_key,
        "ba": round(best_ba, 4),
        "delta_ba": round(best_ba - base_ba, 4),
        "p_value": best_p if best_p is not None else None,
        "significant": bool(best_p is not None and best_p < 0.05),
    }

    # Add metadata
    results["task"] = "age_covariate"
    results["timestamp"] = datetime.datetime.now().isoformat()

    _save_results(results)

    # Generate charts and summary
    print("\nGenerating visualizations and summary...")
    generate_charts(results)
    generate_summary(results)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Age Covariate Integration Complete")
    print(f"  Best strategy: {best_key} (BA={best_ba:.4f}, "
          f"delta={best_ba - base_ba:+.4f}, p={best_p})")
    print(f"  Significant: {best_p is not None and best_p < 0.05}")


if __name__ == "__main__":
    main()
