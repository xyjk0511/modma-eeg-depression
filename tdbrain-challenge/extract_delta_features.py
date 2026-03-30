"""Extract delta-band (1-4 Hz) coherence and asymmetry features.

Computes two feature types per subject per condition (EO, EC):
  - Delta coherence: 12 inter-channel magnitude-squared coherence values
  - Delta asymmetry: 10 hemispheric log-power asymmetry values

Cached to cache_delta/{condition}/{sid}.npz for fast re-runs.

Usage:
  python extract_delta_features.py              # extract all (EO + EC)
  python extract_delta_features.py --condition EO  # single condition
  python extract_delta_features.py --check       # report cache coverage
"""
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import coherence as scipy_coherence
from tqdm import tqdm

from config import CONDITIONS, FREQ_BANDS
from data_loader import load_raw
from main_diagnostic import load_discovery_subjects
from preprocessor import preprocess

CACHE_DIR = Path("cache_delta")
DELTA_FMIN, DELTA_FMAX = FREQ_BANDS["delta"]  # 1, 4

# 12 channel pairs for coherence
COHERENCE_PAIRS = [
    ("F3", "F4"), ("Fp1", "Fp2"), ("F7", "F8"),       # frontal
    ("Fz", "Cz"), ("F3", "C3"), ("F4", "C4"),         # fronto-central
    ("F3", "P3"), ("F4", "P4"),                        # fronto-parietal
    ("C3", "C4"), ("T7", "T8"),                        # inter-hemispheric
    ("P3", "P4"), ("O1", "O2"),                        # parieto-occipital
]

# 10 symmetric pairs for asymmetry (right, left)
ASYMMETRY_PAIRS = [
    ("Fp2", "Fp1"), ("F8", "F7"), ("F4", "F3"), ("FC4", "FC3"),
    ("T8", "T7"), ("C4", "C3"), ("CP4", "CP3"),
    ("P8", "P7"), ("P4", "P3"), ("O2", "O1"),
]

def _compute_delta_coherence(epochs, sfreq):
    """Compute mean delta-band coherence for 12 channel pairs.

    Returns (12,) array. NaN for pairs that fail.
    """
    ch_names = epochs.ch_names
    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    # Average across epochs for coherence stability
    avg_data = data.mean(axis=0)  # (n_ch, n_times)

    coh_values = np.full(len(COHERENCE_PAIRS), np.nan)
    nperseg = int(sfreq * 2)

    for i, (ch_a, ch_b) in enumerate(COHERENCE_PAIRS):
        try:
            idx_a = ch_names.index(ch_a)
            idx_b = ch_names.index(ch_b)
            freqs_c, cxy = scipy_coherence(
                avg_data[idx_a], avg_data[idx_b],
                fs=sfreq, nperseg=nperseg)
            mask = (freqs_c >= DELTA_FMIN) & (freqs_c < DELTA_FMAX)
            if mask.any():
                coh_values[i] = float(np.mean(cxy[mask]))
        except Exception:
            pass  # leave NaN

    return coh_values


def _compute_delta_asymmetry(epochs):
    """Compute delta-band log-power asymmetry for 10 hemispheric pairs.

    asymmetry = ln(delta_power_right) - ln(delta_power_left)
    Returns (10,) array.
    """
    psd = epochs.compute_psd(
        method="welch", fmin=1.0, fmax=40.0, n_fft=500, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)  # (n_epochs, 26, n_freqs)
    ch_names = epochs.ch_names

    delta_mask = (freqs >= DELTA_FMIN) & (freqs < DELTA_FMAX)
    # Mean delta power per channel across epochs
    delta_power = psds[:, :, delta_mask].mean(axis=2).mean(axis=0)  # (26,)

    asym_values = np.full(len(ASYMMETRY_PAIRS), np.nan)
    for i, (ch_right, ch_left) in enumerate(ASYMMETRY_PAIRS):
        try:
            idx_r = ch_names.index(ch_right)
            idx_l = ch_names.index(ch_left)
            pr = delta_power[idx_r]
            pl = delta_power[idx_l]
            if pr > 0 and pl > 0:
                asym_values[i] = float(np.log(pr) - np.log(pl))
        except (ValueError, RuntimeWarning):
            pass  # leave NaN

    return asym_values

def extract_delta_condition(condition):
    """Extract delta features for all DISCOVERY subjects in one condition."""
    cache = CACHE_DIR / condition
    cache.mkdir(parents=True, exist_ok=True)

    subjects = load_discovery_subjects(condition)
    total = len(subjects)
    extracted, skipped, cached = 0, 0, 0

    for _, row in tqdm(subjects.iterrows(), total=total, desc=f"delta-{condition}"):
        sid = row["participants_ID"]
        npz = cache / f"{sid}.npz"

        if npz.exists():
            cached += 1
            continue

        try:
            raw = load_raw(sid, condition)
            epochs, info = preprocess(raw)
        except Exception as e:
            print(f"[SKIP] {sid} {condition}: {e}")
            skipped += 1
            continue

        if len(epochs) == 0:
            print(f"[SKIP] {sid} {condition}: 0 epochs after rejection")
            skipped += 1
            continue

        try:
            sfreq = epochs.info["sfreq"]
            coh = _compute_delta_coherence(epochs, sfreq)
            asym = _compute_delta_asymmetry(epochs)
            np.savez(npz, coherence=coh, asymmetry=asym)
            extracted += 1
        except Exception as e:
            print(f"[ERROR] {sid} {condition} coherence/asymmetry: {e}")
            # Save NaN arrays so we don't retry
            np.savez(npz,
                     coherence=np.full(len(COHERENCE_PAIRS), np.nan),
                     asymmetry=np.full(len(ASYMMETRY_PAIRS), np.nan))
            skipped += 1

    print(f"[{condition}] done: {extracted} extracted, {cached} cached, "
          f"{skipped} skipped, {total} total")


def check_cache():
    """Report cache coverage statistics."""
    for condition in CONDITIONS:
        cache = CACHE_DIR / condition
        subjects = load_discovery_subjects(condition)
        total = len(subjects)
        sids = set(subjects["participants_ID"])

        cached_files = list(cache.glob("*.npz")) if cache.exists() else []
        cached_sids = {f.stem for f in cached_files} & sids
        n_cached = len(cached_sids)

        # Check feature shapes of first cached file
        shape_info = ""
        if cached_files:
            d = np.load(cached_files[0])
            shape_info = (f"  coherence={d['coherence'].shape}, "
                          f"asymmetry={d['asymmetry'].shape}")

        pct = 100 * n_cached / total if total > 0 else 0
        print(f"[{condition}] {n_cached}/{total} ({pct:.1f}%) cached{shape_info}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract delta-band coherence and asymmetry features.")
    parser.add_argument("--condition", choices=CONDITIONS,
                        help="Extract single condition (default: both)")
    parser.add_argument("--check", action="store_true",
                        help="Report cache coverage stats only")
    args = parser.parse_args()

    if args.check:
        check_cache()
        return

    conditions = [args.condition] if args.condition else CONDITIONS
    for cond in conditions:
        extract_delta_condition(cond)

    # Final coverage report
    print("\n--- Cache coverage ---")
    check_cache()


if __name__ == "__main__":
    main()
