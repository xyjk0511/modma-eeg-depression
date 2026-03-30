"""Permutation test for SVM MDD classifier statistical significance.
Runs 1000-iteration group-level label permutation on two feature sets
(v3_subset 676-dim, full 992-dim) with checkpoint support.
"""
import argparse
import datetime
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from run_ablation import load_all, _indices_for_condition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
V3_CATEGORIES = ["abs_bp", "rel_bp", "Hjorth"]
N_SPLITS = 5
BONFERRONI_COMPARISONS = 2
SIGNIFICANCE_THRESHOLD = 0.05 / BONFERRONI_COMPARISONS  # 0.025

RESULTS_PATH = Path("permutation_results.json")
FIGURE_PATH = Path("figures/permutation_null_dist.png")
SUMMARY_PATH = Path("permutation_summary.md")


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def _make_pipe(rs=42):
    """Fixed SVM pipeline (Phase 2 locked decision: class_weight=balanced, no SMOTE)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=rs)),
    ])


# ---------------------------------------------------------------------------
# Observed BA computation
# ---------------------------------------------------------------------------
def _compute_observed_ba(X, y_enc, groups, n_splits=N_SPLITS):
    """Compute observed Balanced Accuracy via 5-fold StratifiedGroupKFold OOF."""
    le = LabelEncoder()
    le.fit(["MDD", "nonMDD"])
    mdd_idx = list(le.classes_).index("MDD")

    cv = StratifiedGroupKFold(n_splits=n_splits)
    preds, test_idxs = [], []

    for train_idx, test_idx in cv.split(X, y_enc, groups):
        pipe = _make_pipe()
        pipe.fit(X[train_idx], y_enc[train_idx])
        preds.append(pipe.predict(X[test_idx]))
        test_idxs.append(test_idx)

    order = np.argsort(np.concatenate(test_idxs))
    pred = np.concatenate(preds)[order]

    y_bin = (y_enc == mdd_idx).astype(int)
    pred_bin = (pred == mdd_idx).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return (sen + spe) / 2


# ---------------------------------------------------------------------------
# Group-level label permutation
# ---------------------------------------------------------------------------
def _permute_labels(y_enc, groups, rng):
    """Permute labels at group level, preserving within-group consistency."""
    unique_groups = np.unique(groups)
    group_label_map = {g: y_enc[groups == g][0] for g in unique_groups}
    group_label_vals = np.array([group_label_map[g] for g in unique_groups])
    perm_map = dict(zip(unique_groups, rng.permutation(group_label_vals)))
    return np.array([perm_map[g] for g in groups])

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _checkpoint_path(feature_set):
    return Path(f"permutation_checkpoint_{feature_set}.json")


def _load_checkpoint(feature_set):
    path = _checkpoint_path(feature_set)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("null_scores", []), data.get("n_done", 0)
    return [], 0


def _save_checkpoint(feature_set, null_scores, n_done, n_total, observed_ba):
    path = _checkpoint_path(feature_set)
    data = {
        "feature_set": feature_set,
        "n_done": n_done,
        "n_total": n_total,
        "observed_ba": observed_ba,
        "null_scores": null_scores,
    }
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _delete_checkpoint(feature_set):
    path = _checkpoint_path(feature_set)
    if path.exists():
        path.unlink()

# ---------------------------------------------------------------------------
# Permutation loop
# ---------------------------------------------------------------------------
def run_permutation_test(X, y_enc, groups, feature_set_name,
                         n_perms=1000, checkpoint_every=100):
    """Run permutation test with checkpoint support.

    Returns (observed_ba, null_scores).
    """
    observed_ba = _compute_observed_ba(X, y_enc, groups)
    print(f"  Observed BA for {feature_set_name}: {observed_ba:.4f}")

    null_scores, n_done = _load_checkpoint(feature_set_name)
    if n_done > 0:
        print(f"  Resuming from checkpoint: {n_done}/{n_perms} done")

    le = LabelEncoder()
    le.fit(["MDD", "nonMDD"])
    mdd_idx = list(le.classes_).index("MDD")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS)

    for i in tqdm(range(n_done, n_perms), desc=f"  {feature_set_name}",
                  initial=n_done, total=n_perms):
        rng = np.random.RandomState(42 + i)
        y_perm = _permute_labels(y_enc, groups, rng)

        preds, test_idxs = [], []
        for train_idx, test_idx in cv.split(X, y_perm, groups):
            pipe = _make_pipe()
            pipe.fit(X[train_idx], y_perm[train_idx])
            preds.append(pipe.predict(X[test_idx]))
            test_idxs.append(test_idx)

        order = np.argsort(np.concatenate(test_idxs))
        pred = np.concatenate(preds)[order]

        y_bin = (y_perm == mdd_idx).astype(int)
        pred_bin = (pred == mdd_idx).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()
        sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        null_scores.append((sen + spe) / 2)

        if (i + 1) % checkpoint_every == 0:
            _save_checkpoint(feature_set_name, null_scores, i + 1,
                             n_perms, observed_ba)

    _delete_checkpoint(feature_set_name)
    return observed_ba, null_scores

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _compute_stats(observed_ba, null_scores, n_perms):
    """Compute p-value, effect size, and CI from null distribution."""
    null_arr = np.array(null_scores)
    # Phipson & Smyth (2010) +1 correction
    p_value = (np.sum(null_arr >= observed_ba) + 1) / (n_perms + 1)
    mean_null = float(np.mean(null_arr))
    std_null = float(np.std(null_arr))
    cohens_d = (observed_ba - mean_null) / std_null if std_null > 0 else float("inf")
    ci_95 = [float(np.percentile(null_arr, 2.5)),
             float(np.percentile(null_arr, 97.5))]
    return {
        "p_value": float(p_value),
        "mean_permuted_ba": round(mean_null, 4),
        "std_permuted_ba": round(std_null, 4),
        "effect_size_cohens_d": round(cohens_d, 4),
        "ci_95": [round(ci_95[0], 4), round(ci_95[1], 4)],
        "significant": bool(p_value < SIGNIFICANCE_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def _save_results(all_results, n_subjects, n_perms):
    output = {
        "task": "permutation_test",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_subjects": n_subjects,
        "n_permutations": n_perms,
        "significance_threshold": SIGNIFICANCE_THRESHOLD,
        "bonferroni_comparisons": BONFERRONI_COMPARISONS,
        "results": all_results,
    }
    with open(RESULTS_PATH, "w", newline="\n", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Null distribution histogram
# ---------------------------------------------------------------------------
def generate_chart(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("figures").mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for ax, (feat_set, r) in zip(axes, all_results.items()):
        null = np.array(r["null_distribution"])
        obs = r["observed_ba"]
        ax.hist(null, bins=50, alpha=0.7, color="#4C72B0", edgecolor="white",
                label="Null distribution")
        ax.axvline(obs, color="red", linestyle="--", linewidth=2,
                   label=f"Observed BA={obs:.4f}")
        sig_str = "significant" if r["significant"] else "not significant"
        ax.text(0.95, 0.95,
                f"p = {r['p_value']:.4f}\n{sig_str}\n(threshold={SIGNIFICANCE_THRESHOLD})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xlabel("Balanced Accuracy")
        ax.set_ylabel("Count")
        ax.set_title(f"{feat_set} ({r['n_features']} features)")
        ax.legend(fontsize=8)

    fig.suptitle(f"Permutation Test Null Distribution ({len(r['null_distribution'])} iterations)")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)
    print(f"Saved {FIGURE_PATH}")

# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def generate_summary(all_results, n_subjects, n_perms):
    lines = []
    lines.append("# Permutation Test Summary\n\n")

    # Section 1: Setup
    lines.append("## 1. Experiment Setup\n\n")
    lines.append(f"- **Model:** SVM (kernel=rbf, class_weight=balanced, random_state=42)\n")
    lines.append(f"- **Pipeline:** StandardScaler + SVC (no SMOTE, Phase 2 locked decision)\n")
    lines.append(f"- **Feature sets:** v3_subset (abs_bp + rel_bp + Hjorth, 676-dim), "
                 f"full (all features, 992-dim)\n")
    lines.append(f"- **Condition:** combined (EO+EC)\n")
    lines.append(f"- **Subjects:** {n_subjects}\n")
    lines.append(f"- **Permutations:** {n_perms}\n")
    lines.append(f"- **CV:** {N_SPLITS}-fold StratifiedGroupKFold (group-level, no subject leakage)\n")
    lines.append(f"- **Permutation level:** Group-level label shuffling\n")
    lines.append(f"- **p-value correction:** Phipson & Smyth (2010) +1 correction\n")
    lines.append(f"- **Multiple comparisons:** Bonferroni correction "
                 f"({BONFERRONI_COMPARISONS} comparisons, threshold p < {SIGNIFICANCE_THRESHOLD})\n\n")

    # Section 2: Results table
    lines.append("## 2. Results\n\n")
    lines.append("| Feature Set | Observed BA | Mean Null BA | Std Null | "
                 "p-value | Significant? | Cohen's d |\n")
    lines.append("|-------------|-------------|-------------|----------|"
                 "---------|-------------|----------|\n")
    for feat_set, r in all_results.items():
        sig = "YES" if r["significant"] else "NO"
        lines.append(
            f"| {feat_set} | {r['observed_ba']:.4f} | {r['mean_permuted_ba']:.4f} "
            f"| {r['std_permuted_ba']:.4f} | {r['p_value']:.4f} | {sig} "
            f"| {r['effect_size_cohens_d']:.4f} |\n"
        )
    lines.append("\n")

    # Section 3: Significance determination
    lines.append("## 3. Significance Determination\n\n")
    lines.append(f"Bonferroni correction applied for {BONFERRONI_COMPARISONS} comparisons: "
                 f"alpha_corrected = 0.05 / {BONFERRONI_COMPARISONS} = {SIGNIFICANCE_THRESHOLD}\n\n")
    for feat_set, r in all_results.items():
        if r["significant"]:
            lines.append(f"- **{feat_set}:** p = {r['p_value']:.4f} < {SIGNIFICANCE_THRESHOLD} "
                         f"-- **statistically significant**. The observed BA ({r['observed_ba']:.4f}) "
                         f"is unlikely under the null hypothesis.\n")
        else:
            lines.append(f"- **{feat_set}:** p = {r['p_value']:.4f} >= {SIGNIFICANCE_THRESHOLD} "
                         f"-- **not statistically significant** at the Bonferroni-corrected level.\n")
    lines.append("\n")
    # Section 4: Power analysis (if any result not significant)
    any_not_sig = any(not r["significant"] for r in all_results.values())
    if any_not_sig:
        lines.append("## 4. Power Analysis Discussion\n\n")
        lines.append(f"One or more feature sets did not reach statistical significance "
                     f"at the Bonferroni-corrected threshold.\n\n")
        for feat_set, r in all_results.items():
            if not r["significant"]:
                d = r["effect_size_cohens_d"]
                lines.append(f"### {feat_set}\n\n")
                lines.append(f"- **Current sample size:** {n_subjects} subjects\n")
                lines.append(f"- **Effect size (Cohen's d):** {d:.4f}\n")
                lines.append(f"- **Observed BA:** {r['observed_ba']:.4f}\n")
                lines.append(f"- **95% CI of null:** [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]\n\n")
                if d > 0 and d < float("inf"):
                    # Approximate sample size for 80% power (z_alpha=1.96, z_beta=0.84)
                    n_needed = int(np.ceil(((1.96 + 0.84) / d) ** 2))
                    lines.append(f"- **Estimated N for 80% power:** ~{n_needed} subjects "
                                 f"(approximate, based on Cohen's d)\n")
                    lines.append(f"- **Recommendation:** Collect additional data to reach "
                                 f"~{n_needed} subjects, or consider pooling with external datasets.\n\n")
                else:
                    lines.append(f"- Effect size is zero or undefined; "
                                 f"the classifier may not outperform chance for this feature set.\n\n")

    # Section 5: 95% CI
    lines.append(f"## {'5' if any_not_sig else '4'}. Null Distribution 95% CI\n\n")
    for feat_set, r in all_results.items():
        lines.append(f"- **{feat_set}:** [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]\n")
    lines.append("\n")

    with open(SUMMARY_PATH, "w", newline="\n", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved {SUMMARY_PATH}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Permutation test for SVM MDD classifier significance."
    )
    parser.add_argument("--n-perms", type=int, default=1000,
                        help="Number of permutations (default: 1000)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Checkpoint interval (default: 100)")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip test; regenerate chart + summary from existing JSON")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (default if checkpoint exists)")
    args = parser.parse_args()

    if args.report_only:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} not found. Run test first.")
            return
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        generate_chart(data["results"])
        generate_summary(data["results"], data["n_subjects"], data["n_permutations"])
        return

    print("Loading cached features...")
    datasets = load_all()
    X_full, y_raw, groups = datasets["combined"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_subjects = len(groups)

    unique, counts = np.unique(y_raw, return_counts=True)
    print(f"  combined: X={X_full.shape}, labels={dict(zip(unique.tolist(), counts.tolist()))}")

    # Feature subsets
    v3_idx = _indices_for_condition(V3_CATEGORIES, "combined")
    feature_sets = {
        "v3_subset": X_full[:, v3_idx],
        "full": X_full,
    }

    all_results = {}
    for feat_name, X_sub in feature_sets.items():
        print(f"\nRunning permutation test: {feat_name} ({X_sub.shape[1]} features)")
        observed_ba, null_scores = run_permutation_test(
            X_sub, y_enc, groups, feat_name,
            n_perms=args.n_perms, checkpoint_every=args.checkpoint_every,
        )
        stats = _compute_stats(observed_ba, null_scores, args.n_perms)
        all_results[feat_name] = {
            "observed_ba": round(observed_ba, 4),
            "n_features": X_sub.shape[1],
            "null_distribution": [round(s, 4) for s in null_scores],
            **stats,
        }
        print(f"  p={stats['p_value']:.4f}, significant={stats['significant']}, "
              f"Cohen's d={stats['effect_size_cohens_d']:.4f}")

    _save_results(all_results, n_subjects, args.n_perms)
    generate_chart(all_results)
    generate_summary(all_results, n_subjects, args.n_perms)

    print(f"\n{'='*60}")
    for feat_name, r in all_results.items():
        sig = "SIGNIFICANT" if r["significant"] else "NOT SIGNIFICANT"
        print(f"  {feat_name}: BA={r['observed_ba']:.4f}, p={r['p_value']:.4f} [{sig}]")


if __name__ == "__main__":
    main()
