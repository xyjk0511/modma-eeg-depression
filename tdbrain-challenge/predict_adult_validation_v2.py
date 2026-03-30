"""Predict on Adult Validation Sample using v3.0 model with feature selection.

This script addresses overfitting by adding SelectKBest feature selection:
- Original: 676 features, Training BA=0.81, CV BA=0.70 (gap=0.11)
- With SelectKBest k=150: Reduced overfitting, better generalization

Usage:
  python predict_adult_validation_v2.py [--k FEATURES]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

from config import PARTICIPANTS_TSV
from preprocessor import preprocess
from feature_extractor import extract_features
from run_ablation import load_all, _indices_for_condition

# Paths
ADULT_VAL_ROOT = Path("AdultValidationSampleTDBRAIN")
CACHE_ADULT_VAL = Path("cache_adult_validation")
RS = 42

# v3.0 configuration
V3_CATEGORIES = ["abs_bp", "rel_bp", "Hjorth"]
HIGH_FP_EXCLUSIONS = {
    "BURNOUT", "CHRONIC PAIN", "INSOMNIA", "OCD",
    "PARKINSON", "SMC", "TINNITUS",
}


# ---------------------------------------------------------------------------
# Feature extraction for Adult Validation subjects
# ---------------------------------------------------------------------------
def extract_adult_validation_features():
    """Extract and cache features for all Adult Validation subjects."""
    subject_dirs = sorted([d for d in ADULT_VAL_ROOT.iterdir()
                          if d.is_dir() and d.name.startswith("sub-")])
    subject_ids = [d.name for d in subject_dirs]
    print(f"Adult Validation subjects: {len(subject_ids)}")

    feats = {}
    for cond in ["EO", "EC"]:
        cache = CACHE_ADULT_VAL / cond
        cache.mkdir(parents=True, exist_ok=True)
        cond_feats = {}

        for i, sid in enumerate(subject_ids):
            npz = cache / f"{sid}.npz"
            if npz.exists():
                d = np.load(npz)
                cond_feats[sid] = d["feat"]
                continue

            task = "restEO" if cond == "EO" else "restEC"
            eeg_file = ADULT_VAL_ROOT / sid / "ses-1" / "eeg" / f"{sid}_ses-1_task-{task}_eeg.vhdr"

            if not eeg_file.exists():
                print(f"[SKIP] {sid} {cond}: file not found")
                continue

            try:
                import mne
                raw = mne.io.read_raw_brainvision(str(eeg_file), preload=True, verbose=False)
                raw.set_montage("standard_1020", on_missing="ignore")
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

            if (i + 1) % 10 == 0:
                print(f"  [{cond}] {i+1}/{len(subject_ids)}")

        feats[cond] = cond_feats
        print(f"  [{cond}] done: {len(cond_feats)}/{len(subject_ids)}")

    common = sorted(set(feats["EO"]) & set(feats["EC"]))
    X_val = np.array([np.concatenate([feats["EO"][s], feats["EC"][s]]) for s in common])
    print(f"Adult Validation combined: {X_val.shape}, {len(common)} subjects")
    return X_val, common


# ---------------------------------------------------------------------------
# Load indication map and apply exclude_high_fp filtering
# ---------------------------------------------------------------------------
def _load_indication_map(subject_ids):
    """Load indication from participants.tsv for given subjects."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    df = df[df["participants_ID"].isin(subject_ids)].drop_duplicates(
        subset=["participants_ID"])
    df["indication_clean"] = df["indication"].fillna("UNKNOWN").astype(str).str.strip()
    df.loc[df["indication_clean"].isin(["nan", ""]), "indication_clean"] = "UNKNOWN"
    return dict(zip(df["participants_ID"], df["indication_clean"]))


def apply_exclude_high_fp(X, y_labels, groups):
    """Apply exclude_high_fp filtering."""
    indication_map = _load_indication_map(list(groups))
    indications = np.array([indication_map.get(g, "UNKNOWN") for g in groups])

    def _should_exclude(ind):
        primary = ind.split("/")[0].strip()
        return primary in HIGH_FP_EXCLUSIONS

    mask = np.array([
        not (_should_exclude(indications[i]) and y_labels[i] == "nonMDD")
        for i in range(len(groups))
    ])

    return X[mask], y_labels[mask], groups[mask]


# ---------------------------------------------------------------------------
# Train with feature selection and validate
# ---------------------------------------------------------------------------
def train_with_feature_selection(X_train, y_train, groups_train, k_features=150):
    """Train model with SelectKBest feature selection and validate."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    mdd_idx = list(le.classes_).index("MDD")

    # Build pipeline with feature selection
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=k_features)),
        ("clf", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=RS)),
    ])

    # Cross-validation to check overfitting
    print(f"\n=== Cross-Validation (k={k_features}) ===")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)

    # CV scores
    cv_scores = []
    for train_idx, val_idx in cv.split(X_train, y_enc, groups_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        pipe_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=k_features)),
            ("clf", SVC(kernel="rbf", probability=True,
                         class_weight="balanced", random_state=RS)),
        ])
        pipe_cv.fit(X_tr, y_tr)
        y_val_pred = pipe_cv.predict(X_val)
        cv_scores.append(balanced_accuracy_score(y_val, y_val_pred))

    cv_ba = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # Train on full data
    pipe.fit(X_train, y_enc)
    y_train_pred = pipe.predict(X_train)
    train_ba = balanced_accuracy_score(y_enc, y_train_pred)

    gap = train_ba - cv_ba

    print(f"Training BA: {train_ba:.4f}")
    print(f"CV BA: {cv_ba:.4f} ± {cv_std:.4f}")
    print(f"Gap: {gap:.4f}")

    if gap < 0.05:
        print("✅ NO OVERFITTING: Gap < 0.05")
    elif gap < 0.10:
        print("⚠️ MILD OVERFITTING: 0.05 < Gap < 0.10")
    else:
        print("❌ SEVERE OVERFITTING: Gap > 0.10")

    # Model complexity
    n_support = len(pipe.named_steps["clf"].support_vectors_)
    print(f"\nSupport vectors: {n_support}/{len(y_enc)} ({100*n_support/len(y_enc):.1f}%)")

    return pipe, mdd_idx, train_ba, cv_ba, gap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict Adult Validation with feature selection")
    parser.add_argument("--k", type=int, default=150,
                        help="Number of features to select (default: 150)")
    parser.add_argument("--search", action="store_true",
                        help="Search for optimal k value")
    args = parser.parse_args()

    # 1. Load DISCOVERY training data
    print("=== Loading DISCOVERY training data ===")
    datasets = load_all()
    X_full, y_full, groups_full = datasets["combined"]
    print(f"DISCOVERY (raw): {X_full.shape}, n={len(groups_full)}")

    # 2. Apply exclude_high_fp filtering
    print("\n=== Applying exclude_high_fp filtering ===")
    X_train, y_train, groups_train = apply_exclude_high_fp(X_full, y_full, groups_full)
    n_mdd = np.sum(y_train == "MDD")
    n_nonmdd = np.sum(y_train == "nonMDD")
    print(f"Filtered: {len(groups_train)} subjects (MDD={n_mdd}, nonMDD={n_nonmdd})")

    # 3. Select v3_subset features
    print("\n=== Selecting v3_subset features ===")
    v3_idx = _indices_for_condition(V3_CATEGORIES, "combined")
    X_train_v3 = X_train[:, v3_idx]
    print(f"v3_subset: {X_train_v3.shape[1]}-dim")

    # 4. Search for optimal k (if requested)
    if args.search:
        print("\n=== Searching for optimal k ===")
        k_values = [50, 100, 150, 200, 250, 300]
        results = []

        for k in k_values:
            _, _, train_ba, cv_ba, gap = train_with_feature_selection(
                X_train_v3, y_train, groups_train, k_features=k)
            results.append({
                "k": k,
                "train_ba": train_ba,
                "cv_ba": cv_ba,
                "gap": gap,
            })
            print()

        # Show results
        print("=== Search Results ===")
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))

        # Find best k (highest CV BA with gap < 0.08)
        valid = df_results[df_results["gap"] < 0.08]
        if len(valid) > 0:
            best_k = valid.loc[valid["cv_ba"].idxmax(), "k"]
            print(f"\n✅ Best k: {best_k}")
        else:
            print("\n⚠️ No k value with gap < 0.08, using k with lowest gap")
            best_k = df_results.loc[df_results["gap"].idxmin(), "k"]
            print(f"Selected k: {best_k}")

        args.k = int(best_k)

    # 5. Train final model
    print(f"\n=== Training final model (k={args.k}) ===")
    pipe, mdd_idx, train_ba, cv_ba, gap = train_with_feature_selection(
        X_train_v3, y_train, groups_train, k_features=args.k)

    # 6. Extract Adult Validation features
    print("\n=== Extracting Adult Validation features ===")
    X_val_full, val_sids = extract_adult_validation_features()
    X_val_v3 = X_val_full[:, v3_idx]
    print(f"Adult Validation v3_subset: {X_val_v3.shape}")

    # 7. Predict
    print("\n=== Generating predictions ===")
    proba = pipe.predict_proba(X_val_v3)[:, mdd_idx]
    pred = np.where(proba >= 0.5, "MDD", "nonMDD")
    n_pos = np.sum(pred == "MDD")
    print(f"Predicted MDD: {n_pos}/{len(pred)} ({100*n_pos/len(pred):.1f}%)")
    print(f"Predicted nonMDD: {len(pred)-n_pos}/{len(pred)} ({100*(len(pred)-n_pos)/len(pred):.1f}%)")

    # 8. Save predictions
    Path("submissions").mkdir(exist_ok=True)

    out_df = pd.DataFrame({
        "participant_id": val_sids,
        "prediction": pred,
    })

    excel_path = f"submissions/adult_validation_predictions_k{args.k}.xlsx"
    csv_path = f"submissions/adult_validation_predictions_k{args.k}.csv"

    out_df.to_excel(excel_path, index=False, sheet_name="Predictions")
    out_df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved {excel_path}")
    print(f"✅ Saved {csv_path}")

    # 9. Summary
    print("\n=== Summary ===")
    print(f"Model: SVM + SelectKBest(k={args.k})")
    print(f"Training: {len(groups_train)} subjects (exclude_high_fp)")
    print(f"Training BA: {train_ba:.4f}")
    print(f"CV BA: {cv_ba:.4f}")
    print(f"Overfitting gap: {gap:.4f}")
    print(f"\nAdult Validation: {len(val_sids)} subjects")
    print(f"Predicted MDD: {n_pos}/{len(val_sids)}")
    print(f"\nSubmit {excel_path} to brainbank@brainclinics.com")


if __name__ == "__main__":
    main()
