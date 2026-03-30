"""Predict on Adult Validation Sample using v3.0 final model.

This script:
1. Extracts features from Adult Validation Sample (60 subjects)
2. Trains v3.0 model on DISCOVERY (exclude_high_fp filtered)
3. Generates discrete predictions (MDD / nonMDD)
4. Saves results in Excel format for submission

Usage:
  python predict_adult_validation.py
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

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
    # Get all subject IDs from directory structure
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

            # Load raw data from Adult Validation Sample
            task = "restEO" if cond == "EO" else "restEC"
            eeg_file = ADULT_VAL_ROOT / sid / "ses-1" / "eeg" / f"{sid}_ses-1_task-{task}_eeg.vhdr"

            if not eeg_file.exists():
                print(f"[SKIP] {sid} {cond}: file not found")
                continue

            try:
                # Load using MNE (Brain Vision format)
                import mne
                raw = mne.io.read_raw_brainvision(str(eeg_file), preload=True, verbose=False)
                # Convert to our standard format
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

    # Build combined array
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
    """Apply exclude_high_fp filtering: remove nonMDD subjects with high-FP indications."""
    indication_map = _load_indication_map(list(groups))
    indications = np.array([indication_map.get(g, "UNKNOWN") for g in groups])

    def _should_exclude(ind):
        primary = ind.split("/")[0].strip()
        return primary in HIGH_FP_EXCLUSIONS

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
# Train on DISCOVERY, predict Adult Validation
# ---------------------------------------------------------------------------
def train_and_predict(X_train, y_train, X_val):
    """Train v3.0 SVM pipeline on filtered DISCOVERY, predict Adult Validation."""
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
    proba = pipe.predict_proba(X_val)[:, mdd_idx]
    pred_label = np.where(proba >= 0.5, "MDD", "nonMDD")

    return proba, pred_label, pipe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict Adult Validation Sample using v3.0 model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate training only (no Adult Validation extraction)")
    args = parser.parse_args()

    # 1. Load DISCOVERY training data
    print("=== Loading DISCOVERY training data ===")
    datasets = load_all()
    X_full, y_full, groups_full = datasets["combined"]
    print(f"DISCOVERY (raw): {X_full.shape}, n={len(groups_full)}")

    # 2. Apply exclude_high_fp filtering
    print("\n=== Applying exclude_high_fp filtering ===")
    X_train, y_train, groups_train = apply_exclude_high_fp(X_full, y_full, groups_full)

    # 3. Select v3_subset features (676d)
    print("\n=== Selecting v3_subset features ===")
    v3_idx = _indices_for_condition(V3_CATEGORIES, "combined")
    X_train_v3 = X_train[:, v3_idx]
    print(f"v3_subset: {X_train_v3.shape[1]}-dim")

    # 4. Dry-run mode: train and exit
    if args.dry_run:
        print("\n=== DRY-RUN MODE ===")
        print(f"Training subjects: {len(groups_train)}")
        print(f"Features: {X_train_v3.shape[1]}")
        print(f"MDD: {np.sum(y_train == 'MDD')}, nonMDD: {np.sum(y_train == 'nonMDD')}")

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

    # 5. Extract Adult Validation features
    print("\n=== Extracting Adult Validation features ===")
    X_val_full, val_sids = extract_adult_validation_features()
    X_val_v3 = X_val_full[:, v3_idx]
    print(f"Adult Validation v3_subset: {X_val_v3.shape}")

    # 6. Train and predict
    print("\n=== Training v3.0 final model ===")
    proba, pred, model = train_and_predict(X_train_v3, y_train, X_val_v3)
    n_pos = np.sum(pred == "MDD")
    print(f"Predicted MDD: {n_pos}/{len(pred)}")

    # 7. Save predictions in Excel format
    Path("submissions").mkdir(exist_ok=True)

    # Create DataFrame with discrete predictions
    out_df = pd.DataFrame({
        "participant_id": val_sids,
        "prediction": pred,
    })

    # Save as Excel (as requested by Prof. Arns)
    excel_path = "submissions/adult_validation_predictions.xlsx"
    out_df.to_excel(excel_path, index=False, sheet_name="Predictions")
    print(f"Saved {excel_path}")

    # Also save CSV for reference
    csv_path = "submissions/adult_validation_predictions.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # 8. Summary
    print("\n=== Summary ===")
    print(f"  DISCOVERY training: {len(groups_train)} subjects (filtered from {X_full.shape[0]})")
    print(f"  Adult Validation: {len(val_sids)} subjects")
    print(f"  Predicted MDD: {n_pos}/{len(val_sids)} ({100*n_pos/len(val_sids):.1f}%)")
    print(f"  Predicted nonMDD: {len(val_sids)-n_pos}/{len(val_sids)} ({100*(len(val_sids)-n_pos)/len(val_sids):.1f}%)")
    print(f"\nSubmit {excel_path} to brainbank@brainclinics.com")


if __name__ == "__main__":
    main()
