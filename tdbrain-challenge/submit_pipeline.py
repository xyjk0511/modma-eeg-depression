"""Submit pipeline: train on DISCOVERY (exclude_high_fp), predict REPLICATION.

v3.0 final model:
  - Training: DISCOVERY subjects with exclude_high_fp filtering (~904 subjects)
  - Features: v3_subset (abs_bp + rel_bp + Hjorth, 676d combined EO+EC)
  - Model: SVM(kernel=rbf, class_weight=balanced, random_state=42)
  - No SMOTE, no SelectKBest

Usage:
  python submit_pipeline.py              # full run: train + predict REPLICATION
  python submit_pipeline.py --dry-run    # validate training only (no REPLICATION extraction)
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from config import PARTICIPANTS_TSV
from data_loader import load_raw
from preprocessor import preprocess
from feature_extractor import extract_features
from run_ablation import load_all, _indices_for_condition

CACHE_REPL = Path("cache_replication")
RS = 42

# v3.0 configuration
V3_CATEGORIES = ["abs_bp", "rel_bp", "Hjorth"]
HIGH_FP_EXCLUSIONS = {
    "BURNOUT", "CHRONIC PAIN", "INSOMNIA", "OCD",
    "PARKINSON", "SMC", "TINNITUS",
}


# ---------------------------------------------------------------------------
# Feature extraction for REPLICATION subjects
# ---------------------------------------------------------------------------
def extract_replication_features():
    """Extract and cache features for all REPLICATION subjects."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    repl = df[df["DISC/REP"] == "REPLICATION"]
    repl_sids = sorted(repl["participants_ID"].unique())
    print(f"REPLICATION subjects: {len(repl_sids)}")

    feats = {}
    for cond in ["EO", "EC"]:
        cache = CACHE_REPL / cond
        cache.mkdir(parents=True, exist_ok=True)
        cond_feats = {}

        for i, sid in enumerate(repl_sids):
            npz = cache / f"{sid}.npz"
            if npz.exists():
                d = np.load(npz)
                cond_feats[sid] = d["feat"]
                continue

            try:
                raw = load_raw(sid, cond)
                epochs, _ = preprocess(raw)
            except Exception as e:
                print(f"[SKIP] {sid} {cond}: {e}")
                continue
            if len(epochs) == 0:
                print(f"[SKIP] {sid} {cond}: 0 epochs")
                continue

            feat = extract_features(epochs)
            np.savez_compressed(npz, feat=feat)
            cond_feats[sid] = feat

            if (i + 1) % 20 == 0:
                print(f"  [{cond}] {i+1}/{len(repl_sids)}")

        feats[cond] = cond_feats
        print(f"  [{cond}] done: {len(cond_feats)}/{len(repl_sids)}")

    # Build combined array
    common = sorted(set(feats["EO"]) & set(feats["EC"]))
    X_repl = np.array([np.concatenate([feats["EO"][s], feats["EC"][s]]) for s in common])
    print(f"REPLICATION combined: {X_repl.shape}, {len(common)} subjects")
    return X_repl, common


# ---------------------------------------------------------------------------
# Indication loading and filtering
# ---------------------------------------------------------------------------
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


def apply_exclude_high_fp(X, y_labels, groups):
    """Apply exclude_high_fp filtering: remove nonMDD subjects with high-FP indications.

    Returns filtered X, y, groups arrays.
    """
    indication_map = _load_indication_map(list(groups))
    indications = np.array([indication_map.get(g, "UNKNOWN") for g in groups])

    def _should_exclude(ind):
        """Check if a nonMDD indication should be excluded."""
        primary = ind.split("/")[0].strip()
        return primary in HIGH_FP_EXCLUSIONS

    # Only exclude nonMDD subjects; keep all MDD subjects
    mask = np.array([
        not (_should_exclude(indications[i]) and y_labels[i] == "nonMDD")
        for i in range(len(groups))
    ])

    X_filtered = X[mask]
    y_filtered = y_labels[mask]
    g_filtered = groups[mask]

    n_mdd = int(np.sum(y_filtered == "MDD"))
    n_nonmdd = int(np.sum(y_filtered == "nonMDD"))
    print(f"exclude_high_fp: {len(g_filtered)} subjects (MDD={n_mdd}, nonMDD={n_nonmdd})")

    return X_filtered, y_filtered, g_filtered


# ---------------------------------------------------------------------------
# Train on DISCOVERY, predict REPLICATION
# ---------------------------------------------------------------------------
def train_and_predict(X_train, y_train, X_repl):
    """Train v3.0 SVM pipeline on filtered DISCOVERY, predict REPLICATION."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    mdd_idx = list(le.classes_).index("MDD")

    # Build v3.0 pipeline: StandardScaler + SVM(rbf, class_weight=balanced)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=RS)),
    ])
    pipe.fit(X_train, y_enc)

    # Predict
    proba = pipe.predict_proba(X_repl)[:, mdd_idx]
    pred_label = np.where(proba >= 0.5, "MDD", "nonMDD")

    return proba, pred_label, pipe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="v3.0 submit pipeline: exclude_high_fp + v3_subset + SVM-only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate training only (no REPLICATION extraction)")
    args = parser.parse_args()

    # 1. Load DISCOVERY training data
    print("Loading DISCOVERY features...")
    datasets = load_all()
    X_full, y_full, groups_full = datasets["combined"]
    print(f"DISCOVERY (raw): {X_full.shape}, n={len(groups_full)}")

    # 2. Apply exclude_high_fp filtering
    print("\nApplying exclude_high_fp filtering...")
    X_train, y_train, groups_train = apply_exclude_high_fp(X_full, y_full, groups_full)

    # 3. Select v3_subset features (676d)
    print("\nSelecting v3_subset features...")
    v3_idx = _indices_for_condition(V3_CATEGORIES, "combined")
    X_train_v3 = X_train[:, v3_idx]
    print(f"v3_subset: {X_train_v3.shape[1]}-dim")

    # 4. Dry-run mode: train and exit
    if args.dry_run:
        print("\n=== DRY-RUN MODE ===")
        print(f"Training subjects: {len(groups_train)}")
        print(f"Features: {X_train_v3.shape[1]}")
        print(f"MDD: {np.sum(y_train == 'MDD')}, nonMDD: {np.sum(y_train == 'nonMDD')}")

        # Train model to validate pipeline
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                         class_weight="balanced", random_state=RS)),
        ])
        pipe.fit(X_train_v3, y_enc)
        print(f"Model trained: {pipe.named_steps['clf']}")
        print("Dry-run complete. Pipeline validated.")
        return

    # 5. Extract REPLICATION features
    print("\nExtracting REPLICATION features...")
    X_repl_full, repl_sids = extract_replication_features()
    X_repl_v3 = X_repl_full[:, v3_idx]
    print(f"REPLICATION v3_subset: {X_repl_v3.shape}")

    # 6. Train and predict
    print("\n=== Training v3.0 final model ===")
    proba, pred, model = train_and_predict(X_train_v3, y_train, X_repl_v3)
    n_pos = np.sum(pred == "MDD")
    print(f"Predicted MDD: {n_pos}/{len(pred)}")

    # 7. Save predictions
    Path("submissions").mkdir(exist_ok=True)
    out_df = pd.DataFrame({
        "participants_ID": repl_sids,
        "prediction": pred,
        "probability_MDD": np.round(proba, 4),
    })
    csv_path = "submissions/prediction_v3_svm.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # 8. Summary
    summary = {
        "task": "MDD_diagnostic_REPLICATION",
        "version": "v3.0",
        "model": "SVM(kernel=rbf, class_weight=balanced)",
        "features": "v3_subset (abs_bp + rel_bp + Hjorth, 676d)",
        "filtering": "exclude_high_fp (BURNOUT, CHRONIC PAIN, INSOMNIA, OCD, PARKINSON, SMC, TINNITUS)",
        "n_discovery_raw": int(X_full.shape[0]),
        "n_discovery_filtered": int(len(groups_train)),
        "n_replication": len(repl_sids),
        "n_predicted_MDD": int(n_pos),
        "threshold": 0.5,
    }
    with open("submissions/summary.json", "w", newline="\n") as f:
        json.dump(summary, f, indent=2)
    print("Saved submissions/summary.json")

    print("\n=== Done ===")
    print(f"  DISCOVERY: {len(groups_train)} subjects (filtered from {X_full.shape[0]})")
    print(f"  REPLICATION: {len(repl_sids)} subjects")
    print(f"  Predicted MDD: {n_pos}/{len(repl_sids)}")


if __name__ == "__main__":
    main()
