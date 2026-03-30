"""Optimization evaluation: compare improvements over baseline.
Tests: ses-2 averaging, demographics, calibration, LightGBM, stacking.
"""
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import DATASET_ROOT, PARTICIPANTS_TSV, CONDITIONS
from data_loader import load_raw
from preprocessor import preprocess
from feature_extractor import extract_features
from run_ablation import _indices_for_condition

CACHE_DIR = Path("cache_diagnostic")
CACHE_SES2 = Path("cache_diagnostic_ses2")
RS = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_cached(cache_path):
    """Load {sid: (feat, label)} from a cache directory."""
    feats = {}
    for npz in cache_path.glob("*.npz"):
        d = np.load(npz)
        feats[npz.stem] = (d["feat"], str(d["label"]))
    return feats


def _get_discovery_sids(condition):
    """Return set of valid DISCOVERY subject IDs for a condition."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    mask = (df["DISC/REP"] == "DISCOVERY") & (df[condition] == 1)
    if condition == "EO":
        mask = mask & ~(df["participants_ID"] == "sub-88049673")
    return set(df[mask]["participants_ID"].unique())


def _get_multi_session_sids():
    """Return set of subject IDs that have nrSessions == 2."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    return set(df[df["nrSessions"] == 2]["participants_ID"].unique())


def extract_ses2_features(condition, sids_with_ses2, labels_map):
    """Extract and cache ses-2 features for multi-session subjects."""
    cache = CACHE_SES2 / condition
    cache.mkdir(parents=True, exist_ok=True)
    feats = {}
    for sid in sorted(sids_with_ses2):
        npz = cache / f"{sid}.npz"
        if npz.exists():
            d = np.load(npz)
            feats[sid] = (d["feat"], str(d["label"]))
            continue
        try:
            raw = load_raw(sid, condition, session=2)
            epochs, _ = preprocess(raw)
        except Exception as e:
            print(f"[SKIP ses-2] {sid} {condition}: {e}")
            continue
        if len(epochs) == 0:
            continue
        feat = extract_features(epochs)
        label = labels_map.get(sid, "nonMDD")
        np.savez_compressed(npz, feat=feat, label=np.array(label))
        feats[sid] = (feat, label)
    print(f"  [ses-2 {condition}] {len(feats)}/{len(sids_with_ses2)} extracted")
    return feats


def load_demographics():
    """Return {sid: (age_float, gender_int)} for all subjects."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    df = df.drop_duplicates("participants_ID")
    demo = {}
    ages = df["age"].str.replace(",", ".").astype(float)
    median_age = ages.median()
    for _, row in df.iterrows():
        sid = row["participants_ID"]
        age_str = row["age"]
        age = float(age_str.replace(",", ".")) if pd.notna(age_str) else median_age
        gender = float(row["gender"]) if pd.notna(row["gender"]) else 0.0
        demo[sid] = (age, gender)
    return demo, median_age


def build_dataset(use_ses2=False, use_demo=False, categories=None):
    """Build combined feature matrix with optional improvements.
    Returns X, y, groups, sids.
    """
    sids_eo = _get_discovery_sids("EO")
    sids_ec = _get_discovery_sids("EC")
    feats_eo = {s: v for s, v in _load_cached(CACHE_DIR / "EO").items() if s in sids_eo}
    feats_ec = {s: v for s, v in _load_cached(CACHE_DIR / "EC").items() if s in sids_ec}
    common = sorted(sids_eo & sids_ec & set(feats_eo) & set(feats_ec))

    # Session averaging for ses-2
    if use_ses2:
        multi_sids = _get_multi_session_sids()
        labels_map = {s: feats_eo[s][1] for s in common if s in feats_eo}
        ses2_eo = extract_ses2_features("EO", multi_sids & set(common), labels_map)
        ses2_ec = extract_ses2_features("EC", multi_sids & set(common), labels_map)
        for sid in common:
            if sid in ses2_eo:
                avg = (feats_eo[sid][0] + ses2_eo[sid][0]) / 2.0
                feats_eo[sid] = (avg, feats_eo[sid][1])
            if sid in ses2_ec:
                avg = (feats_ec[sid][0] + ses2_ec[sid][0]) / 2.0
                feats_ec[sid] = (avg, feats_ec[sid][1])

    # Build combined EEG feature vector
    X_eeg = np.array([np.concatenate([feats_eo[s][0], feats_ec[s][0]]) for s in common])
    y = np.array([feats_eo[s][1] for s in common])

    # Feature subset selection
    if categories:
        idx = _indices_for_condition(categories, "combined")
        X_eeg = X_eeg[:, idx]

    # Append demographics
    if use_demo:
        demo, _ = load_demographics()
        demo_arr = np.array([[demo[s][0], demo[s][1]] for s in common])
        X_eeg = np.hstack([X_eeg, demo_arr])

    return X_eeg, y, np.array(common), common
