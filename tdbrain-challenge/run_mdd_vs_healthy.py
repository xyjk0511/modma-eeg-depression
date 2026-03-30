"""MDD-vs-Healthy reframing experiment.
Tests whether filtering out psychiatric nonMDD subjects improves classification
by removing confounding EEG patterns that overlap with MDD.

Experiments:
  1. full_baseline: All 1138 subjects (MDD=305, nonMDD=833)
  2. mdd_vs_healthy: Only exact MDD + exact HEALTHY subjects
  3. exclude_high_fp: Remove 7 high-FP psychiatric groups

Usage:
  python run_mdd_vs_healthy.py            # run all experiments
  python run_mdd_vs_healthy.py --report-only  # regenerate charts/summary from JSON
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

# 7 high-FP nonMDD subgroups from Phase 4 error analysis
HIGH_FP_EXCLUSIONS = {
    "BURNOUT", "CHRONIC PAIN", "INSOMNIA", "OCD",
    "PARKINSON", "SMC", "TINNITUS",
}

RESULTS_PATH = Path("mdd_reframing_results.json")
SUMMARY_PATH = Path("mdd_reframing_summary.md")
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


def _load_indication_map(subject_ids):
    """Load indication from participants.tsv for given subjects.
    Subjects with NaN indication are mapped to 'UNKNOWN'."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    df = df[df["participants_ID"].isin(subject_ids)].drop_duplicates(
        subset=["participants_ID"])
    df["indication_clean"] = df["indication"].fillna("UNKNOWN").astype(str).str.strip()
    # Normalize "nan" string to "UNKNOWN"
    df.loc[df["indication_clean"].isin(["nan", ""]), "indication_clean"] = "UNKNOWN"
    return dict(zip(df["participants_ID"], df["indication_clean"]))


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


def _independent_permutation_test(fold_ba_exp, fold_ba_base):
    """Permutation test on independent fold BA arrays (different fold compositions)."""
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
# Data validation and filtering
# ---------------------------------------------------------------------------
def validate_indications(indication_map, groups):
    """Validate indication data for all subjects."""
    missing = [g for g in groups if g not in indication_map]
    assert len(missing) == 0, (
        f"{len(missing)} subjects not found in participants.tsv: {missing[:5]}")

    # Count UNKNOWN indications
    unknown_count = sum(1 for g in groups if indication_map[g] == "UNKNOWN")
    print(f"\nSubjects with unknown indication: {unknown_count} "
          f"(labeled nonMDD, excluded from filtered strategies)")

    # Log unique indications
    from collections import Counter
    counts = Counter(indication_map[g] for g in groups)
    print("\nIndication distribution (combined dataset):")
    for ind, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {ind}: {cnt}")

    # Assert known indications exist
    ind_set = set(counts.keys())
    for required in ["MDD", "HEALTHY", "ADHD"]:
        assert required in ind_set, f"Missing expected indication: {required}"

    return counts


def build_strategies(groups, indication_map, y_labels):
    """Build boolean masks for each filtering strategy.

    Returns dict of {strategy_name: (mask, description, composition)}.
    """
    n = len(groups)
    indications = np.array([indication_map[g] for g in groups])

    # Strategy 1: full_baseline — all subjects
    mask_full = np.ones(n, dtype=bool)

    # Strategy 2: mdd_vs_healthy — exact MDD + exact HEALTHY only
    mask_mdd_healthy = np.array([
        ind == "MDD" or ind == "HEALTHY" for ind in indications
    ])

    # Strategy 3: exclude_high_fp — remove 7 psychiatric groups
    # For compound nonMDD indications, exclude if primary (before "/") is in list
    def _should_exclude(ind):
        """Check if a nonMDD indication should be excluded."""
        primary = ind.split("/")[0].strip()
        return primary in HIGH_FP_EXCLUSIONS
    # Only exclude nonMDD subjects; keep all MDD subjects (including MDD/*)
    mask_exclude = np.array([
        not (_should_exclude(ind) and y_labels[i] == "nonMDD")
        for i, ind in enumerate(indications)
    ])

    # Log mixed MDD diagnoses
    mdd_mixed = [ind for ind in indications if ind.startswith("MDD/")]
    print(f"\nMixed MDD diagnoses (MDD/*): {len(mdd_mixed)} subjects "
          f"(excluded from mdd_vs_healthy strategy)")

    strategies = {}
    for name, mask in [("full_baseline", mask_full),
                       ("mdd_vs_healthy", mask_mdd_healthy),
                       ("exclude_high_fp", mask_exclude)]:
        y_sub = y_labels[mask]
        n_mdd = int(np.sum(y_sub == "MDD"))
        n_nonmdd = int(np.sum(y_sub == "nonMDD"))
        # Indications included
        inds_included = sorted(set(indications[mask]))
        strategies[name] = {
            "mask": mask,
            "n_subjects": int(mask.sum()),
            "n_mdd": n_mdd,
            "n_nonmdd": n_nonmdd,
            "indications_included": inds_included,
        }
        print(f"\n{name}: n={mask.sum()} (MDD={n_mdd}, nonMDD={n_nonmdd})")

    return strategies

# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def run_strategy(name, X_v3, y_enc, groups, mdd_idx, strategy_info):
    """Run CV evaluation for a single filtering strategy."""
    mask = strategy_info["mask"]
    X_sub = X_v3[mask]
    y_sub = y_enc[mask]
    g_sub = groups[mask]

    print(f"\n[{name}] n={len(g_sub)} "
          f"(MDD={strategy_info['n_mdd']}, nonMDD={strategy_info['n_nonmdd']})")
    metrics, probas, y_bin = _cv_evaluate(
        X_sub, y_sub, g_sub, mdd_idx, desc=name)
    metrics["n_subjects"] = strategy_info["n_subjects"]
    metrics["n_mdd"] = strategy_info["n_mdd"]
    metrics["n_nonmdd"] = strategy_info["n_nonmdd"]
    metrics["indications_included"] = strategy_info["indications_included"]
    print(f"  BA={metrics['ba']:.4f}, AUC={metrics['auc']:.4f}, "
          f"SEN={metrics['sen']:.4f}, SPE={metrics['spe']:.4f}")
    return metrics, probas, y_bin


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_charts(results):
    """Generate BA comparison bar chart and ROC curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)
    experiments = results.get("experiments", {})

    # --- 1. BA comparison bar chart ---
    strategies = ["full_baseline", "mdd_vs_healthy", "exclude_high_fp"]
    labels = ["Full Dataset\n(n={})".format(
                  experiments.get("full_baseline", {}).get("n_subjects", "?")),
              "MDD vs Healthy\n(n={})".format(
                  experiments.get("mdd_vs_healthy", {}).get("n_subjects", "?")),
              "Exclude High-FP\n(n={})".format(
                  experiments.get("exclude_high_fp", {}).get("n_subjects", "?"))]
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
    ax.axhline(0.6318, color="gray", linestyle="--", linewidth=1.5,
               label="Pooled baseline BA=0.6318")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("MDD-vs-Healthy Reframing: BA Comparison")
    ax.legend(fontsize=9)

    for bar, ba_val, std_val in zip(bars, bas, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std_val + 0.005,
                f"{ba_val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.5, max(bas) + 0.08)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "mdd_reframing_ba_comparison.png")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'mdd_reframing_ba_comparison.png'}")

    # --- 2. ROC curves ---
    roc_data = results.get("roc_data", {})
    if roc_data:
        fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)
        colors_roc = {"full_baseline": "#1f77b4",
                      "mdd_vs_healthy": "#2ca02c",
                      "exclude_high_fp": "#ff7f0e"}
        for s in strategies:
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
        ax2.set_title("MDD-vs-Healthy Reframing: ROC Curves")
        ax2.legend(fontsize=9, loc="lower right")
        fig2.tight_layout()
        fig2.savefig(FIGURE_DIR / "mdd_reframing_roc.png")
        plt.close(fig2)
        print(f"Saved {FIGURE_DIR / 'mdd_reframing_roc.png'}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def generate_summary(results):
    """Generate human-readable markdown summary."""
    experiments = results.get("experiments", {})
    stats = results.get("statistical_tests", {})
    baseline = experiments.get("full_baseline", {})
    lines = []
    lines.append("# MDD-vs-Healthy Reframing Summary\n\n")

    # Section 1: Setup
    lines.append("## 1. Experiment Setup\n\n")
    lines.append("- **Pipeline:** StandardScaler + SVC(kernel=rbf, "
                 "class_weight=balanced, random_state=42)\n")
    lines.append("- **Base features:** v3_subset (abs_bp + rel_bp + Hjorth, "
                 "676-dim, combined EO+EC)\n")
    lines.append(f"- **CV:** {N_SPLITS}-fold StratifiedGroupKFold\n")
    lines.append("- **Statistical test:** scipy permutation_test, "
                 "permutation_type=independent, n_resamples=9999\n\n")

    # Section 2: Dataset composition
    lines.append("## 2. Dataset Composition\n\n")
    lines.append("| Strategy | Total | MDD | nonMDD | Description |\n")
    lines.append("|----------|-------|-----|--------|-------------|\n")
    for s, desc in [("full_baseline", "All DISCOVERY subjects"),
                    ("mdd_vs_healthy", "Exact MDD + exact HEALTHY only"),
                    ("exclude_high_fp", "Remove 7 high-FP psychiatric groups")]:
        r = experiments.get(s, {})
        lines.append(f"| {s} | {r.get('n_subjects', '?')} | "
                     f"{r.get('n_mdd', '?')} | {r.get('n_nonmdd', '?')} | "
                     f"{desc} |\n")
    lines.append("\n")

    # Section 3: Results table
    lines.append("## 3. Results Comparison\n\n")
    lines.append("| Strategy | BA | AUC | SEN | SPE | delta-BA | p-value |\n")
    lines.append("|----------|-----|-----|-----|-----|----------|--------|\n")

    base_ba = baseline.get("ba", 0)
    lines.append(f"| full_baseline | {base_ba:.4f} | "
                 f"{baseline.get('auc', 0):.4f} | "
                 f"{baseline.get('sen', 0):.4f} | "
                 f"{baseline.get('spe', 0):.4f} | - | - |\n")

    for s in ["mdd_vs_healthy", "exclude_high_fp"]:
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

    # Section 4: Statistical tests
    lines.append("## 4. Statistical Tests\n\n")
    for s in ["mdd_vs_healthy", "exclude_high_fp"]:
        st = stats.get(f"{s}_vs_baseline", {})
        lines.append(f"- **{s} vs full_baseline:** "
                     f"delta-BA={st.get('delta_ba', 0):+.4f}, "
                     f"p={st.get('p_value', 'N/A')}\n")
    lines.append("\n")

    # Section 5: Decision
    lines.append("## 5. Decision & Recommendation\n\n")
    decision = results.get("decision", {})
    lines.append(f"- **Recommended strategy:** "
                 f"{decision.get('recommended', 'N/A')}\n")
    lines.append(f"- **Rationale:** {decision.get('rationale', 'N/A')}\n")
    lines.append(f"- **Adopt for v3.0:** "
                 f"{decision.get('adopt_filtered', 'N/A')}\n\n")

    # Section 6: Caveats
    lines.append("## 6. Caveats\n\n")
    lines.append("- mdd_vs_healthy has far fewer nonMDD subjects (45 HEALTHY "
                 "vs 833 nonMDD), so higher BA may partly reflect an easier "
                 "task rather than a better model\n")
    lines.append("- exclude_high_fp is a more conservative approach that "
                 "retains more subjects while removing known confounders\n")
    lines.append("- The competition evaluation uses the full nonMDD pool, "
                 "so filtered training may not generalize to the test set\n")

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
    return {"experiments": {}, "statistical_tests": {}, "roc_data": {}}


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
        description="MDD-vs-Healthy reframing experiments.")
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

    # Load indication data
    indication_map = _load_indication_map(list(groups))

    # Validate indications
    ind_counts = validate_indications(indication_map, groups)

    # Build filtering strategies
    strategies = build_strategies(groups, indication_map, y_raw)

    # Check for existing results (checkpoint/resume)
    results = _load_results()
    if "experiments" not in results:
        results["experiments"] = {}
    if "roc_data" not in results:
        results["roc_data"] = {}

    # Run each strategy
    roc_store = {}
    for name, info in strategies.items():
        if name in results["experiments"]:
            print(f"\n[{name}] already done, skipping")
            continue
        metrics, probas, y_bin = run_strategy(
            name, X_v3, y_enc, groups, mdd_idx, info)
        results["experiments"][name] = metrics

        # Store ROC data
        fpr, tpr, _ = roc_curve(y_bin, probas)
        roc_store[name] = {
            "fpr": [round(float(v), 6) for v in fpr],
            "tpr": [round(float(v), 6) for v in tpr],
        }
        results["roc_data"][name] = roc_store[name]
        _save_results(results)

    # --- Statistical comparison ---
    print("\n[Statistical comparison]")
    base_fold_ba = results["experiments"]["full_baseline"]["per_fold_ba"]
    results["statistical_tests"] = {}

    for name in ["mdd_vs_healthy", "exclude_high_fp"]:
        exp = results["experiments"][name]
        fold_ba = exp.get("per_fold_ba", [])
        ba = exp["ba"]
        base_ba = results["experiments"]["full_baseline"]["ba"]
        delta = ba - base_ba

        p = _independent_permutation_test(fold_ba, base_fold_ba)
        results["statistical_tests"][f"{name}_vs_baseline"] = {
            "delta_ba": round(delta, 4),
            "p_value": p,
            "significant": bool(p < 0.05),
        }
        print(f"  {name}: delta-BA={delta:+.4f}, p={p:.6f}")

    # --- Decision logic ---
    mdd_healthy = results["experiments"]["mdd_vs_healthy"]
    exclude_hp = results["experiments"]["exclude_high_fp"]
    base_ba = results["experiments"]["full_baseline"]["ba"]
    mh_test = results["statistical_tests"]["mdd_vs_healthy_vs_baseline"]
    eh_test = results["statistical_tests"]["exclude_high_fp_vs_baseline"]

    if mh_test["significant"] or eh_test["significant"]:
        # At least one filtered strategy is significantly better
        if mh_test["significant"] and eh_test["significant"]:
            best = ("mdd_vs_healthy" if mdd_healthy["ba"] >= exclude_hp["ba"]
                    else "exclude_high_fp")
        elif mh_test["significant"]:
            best = "mdd_vs_healthy"
        else:
            best = "exclude_high_fp"
        rationale = (f"{best} significantly improves BA over full dataset "
                     f"(p<0.05). Psychiatric nonMDD confirmed as confound.")
        adopt = True
    else:
        best = "full_baseline"
        rationale = ("No filtered strategy significantly improves BA (p>=0.05). "
                     "Mixed nonMDD pool is acceptable. Keep full dataset.")
        adopt = False

    results["decision"] = {
        "recommended": best,
        "rationale": rationale,
        "adopt_filtered": adopt,
    }

    # Add metadata
    results["task"] = "mdd_reframing"
    results["timestamp"] = datetime.datetime.now().isoformat()
    _save_results(results)

    # Generate charts and summary
    print("\nGenerating visualizations and summary...")
    generate_charts(results)
    generate_summary(results)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"MDD-vs-Healthy Reframing Complete")
    print(f"  full_baseline: BA={base_ba:.4f}")
    print(f"  mdd_vs_healthy: BA={mdd_healthy['ba']:.4f}, "
          f"delta={mdd_healthy['ba'] - base_ba:+.4f}, "
          f"p={mh_test['p_value']}")
    print(f"  exclude_high_fp: BA={exclude_hp['ba']:.4f}, "
          f"delta={exclude_hp['ba'] - base_ba:+.4f}, "
          f"p={eh_test['p_value']}")
    print(f"  Decision: {best} ({'adopt' if adopt else 'keep full dataset'})")


if __name__ == "__main__":
    main()
