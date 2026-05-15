# -*- coding: utf-8 -*-
"""
=============================================================================
COE216 Signals and Systems - Final Project
Emo Challenge 2026 | Phase 2: Research & Development

Group 9
  - Ahmet Akin
  - Berivan Demir
  - Mustafa Talha Akgul

Istanbul University - Computer Engineering Department

Description:
    Phase 2 upgrades upon the Phase 1 baseline (51-dim / RandomForest) with:

    1. EXPANDED FEATURE SET  (51 → 135 dims)
       Phase 1 features (51 dims) PLUS:
         - MFCC Delta coefficients      (26 dims) — temporal dynamics
         - Chroma STFT                  (24 dims) — harmonic / tonal content
         - Spectral Flatness            ( 2 dims) — noise vs. tonal ratio
         - Mel-Spectrogram statistics   (20 dims) — perceptual frequency scale
         - Tonnetz                      (12 dims) — tonal centroid (harmony)

    2. THREE CLASSIFIERS
         A. RandomForestClassifier      (Phase 1 baseline, upgraded)
         B. XGBoostClassifier           (gradient boosting)
         C. MLPClassifier               (multi-layer perceptron / neural net)

    3. SOFT-VOTING ENSEMBLE
         Combines probability outputs of all three models for final prediction.

    4. HYPERPARAMETER OPTIMISATION
         RandomizedSearchCV on each base model independently.

    5. ERROR ANALYSIS
         Per-class precision/recall heatmap + misclassification pair ranking
         to guide Phase 3 "hard confusion" mitigation strategy.

Dataset Layout:
    Midterm_Dataset_2026/
        <group_folder>/
            *.wav
=============================================================================
"""

# ─────────────────────────── Standard Imports ───────────────────────────────
import os
import sys
import time
import warnings

# Windows'ta dosya isimlerindeki özel karakterleri yazdırırken oluşan Unicode hatasını (charmap codec) önlemek için:
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not installed. Run: pip install xgboost")
    print("       XGBoost model will be skipped.\n")

warnings.filterwarnings("ignore")


# ===========================================================================
# SECTION 1 — CONFIGURATION
# ===========================================================================

DATASET_ROOT    = "Midterm_Dataset_2026"
EMOTION_CLASSES = ["neutral", "happy", "angry", "sad", "surprised"]

EMOTION_MAPPING = {
    "notr": "neutral",   "mutlu": "happy",      "ofkeli": "angry",
    "uzgun": "sad",      "saskin": "surprised",  "sasirma": "surprised",
    "notr": "neutral",   "ofkeli": "angry",      "uzgun": "sad",
    "saskin": "surprised", "sasirma": "surprised",
    # Turkish with special chars
    "nötr": "neutral",   "öfkeli": "angry",      "üzgün": "sad",
    "şaşkın": "surprised", "şaşırma": "surprised",
    # English
    "neutral": "neutral", "happy": "happy",      "angry": "angry",
    "furious": "angry",   "sad": "sad",           "surprised": "surprised",
    "shocked": "surprised",
}

SAMPLE_RATE       = 22050
FRAME_DURATION_MS = 25
HOP_DURATION_MS   = 10

# ── Phase 2 Feature Config ───────────────────────────────────────────────────
N_MFCC      = 13    # base MFCC; delta uses same count
N_CHROMA    = 12    # chroma bins (semitones in one octave)
N_MEL_BANDS = 10    # mel filterbank bands for stat pooling

# ── Output Files ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
OUTPUT_CSV   = "group9_phase2_results.csv"
OUTPUT_DIR   = "."   # all PNGs saved here


# ===========================================================================
# SECTION 2 — DATASET LOADING  (unchanged from Phase 1)
# ===========================================================================

def load_dataset(dataset_root=DATASET_ROOT, emotion_classes=EMOTION_CLASSES):
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset root not found: '{dataset_root}'\n"
            "Please verify DATASET_ROOT."
        )
    file_paths, labels = [], []
    for group_folder in os.listdir(dataset_root):
        group_path = os.path.join(dataset_root, group_folder)
        if not os.path.isdir(group_path):
            continue
        print(f"  [Dataset] Scanning: {group_folder}")
        for filename in os.listdir(group_path):
            if not filename.lower().endswith(".wav"):
                continue
            emotion = _emotion_from_filename(filename)
            if emotion and emotion in emotion_classes:
                file_paths.append(os.path.join(group_path, filename))
                labels.append(emotion)
            else:
                print(f"    [SKIP] Unknown emotion: {filename}")
    print(f"\n[Dataset] Total: {len(file_paths)} files | "
          f"{len(set(labels))} classes.\n")
    return file_paths, labels


def _emotion_from_filename(filename):
    name  = os.path.splitext(filename)[0]
    parts = name.lower().split("_")
    for part in parts:
        key = part.strip().rstrip(".")
        if key in EMOTION_MAPPING:
            return EMOTION_MAPPING[key]
    return None


# ===========================================================================
# SECTION 3 — FEATURE EXTRACTION  (Phase 2 — 135 dims)
# ===========================================================================
#
#   PHASE 1 (51 dims):
#     MFCC mean+std    (26) | STE/ZCR (3) | F0 (2) | Spectral (20)
#
#   PHASE 2 ADDITIONS (84 dims):
#     MFCC Delta       (26) | Chroma (24) | Flatness (2)
#     Mel-Spec stats   (20) | Tonnetz (12)
#
#   TOTAL = 51 + 84 = 135 dims
# ===========================================================================

def _ms_to_samples(ms, sr):
    return int(sr * ms / 1000)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_audio(audio, sr=SAMPLE_RATE):
    """
    1. preemphasis (coef=0.97)  — amplifies fricatives / high-freq consonants
    2. trim silence (top_db=20) — focus on voiced regions only
    """
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    return audio


# ── Phase 1 Sub-extractors (kept identical) ───────────────────────────────────

def _mfcc_features(audio, sr, n_mfcc=N_MFCC):
    """MFCC mean + std → 26 dims"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def _ste_zcr(audio, frame_length, hop_length):
    """Short-Time Energy + ZCR → 3 dims"""
    frames = librosa.util.frame(audio, frame_length=frame_length,
                                 hop_length=hop_length)
    ste = np.sum(frames ** 2, axis=0)
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    return float(np.mean(ste)), float(np.mean(zcr)), float(np.std(zcr))


def _pitch_features(audio, sr):
    """Autocorrelation-based F0 → 2 dims"""
    fl      = _ms_to_samples(FRAME_DURATION_MS, sr)
    hl      = _ms_to_samples(HOP_DURATION_MS, sr)
    frames  = librosa.util.frame(audio, frame_length=fl, hop_length=hl)
    lag_min = max(1, int(sr / 500))
    lag_max = int(sr / 50)
    f0_vals = []
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        r = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if lag_max >= len(r) or r[0] < 1e-12:
            continue
        best_lag   = int(np.argmax(r[lag_min: lag_max + 1])) + lag_min
        confidence = r[best_lag] / r[0]
        if confidence >= 0.3:
            f0_vals.append(float(sr) / float(best_lag))
    return (float(np.mean(f0_vals)), float(np.std(f0_vals))) \
        if f0_vals else (0.0, 0.0)


def _spectral_features(audio, sr):
    """Centroid/BW/Rolloff (6) + Contrast (14) → 20 dims"""
    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    contrast  = librosa.feature.spectral_contrast(y=audio, sr=sr)
    feats = np.array([
        np.mean(centroid),  np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff),   np.std(rolloff),
    ])
    return np.concatenate([feats,
                            np.mean(contrast, axis=1),
                            np.std(contrast,  axis=1)])   # 20


# ── Phase 2 New Sub-extractors ────────────────────────────────────────────────

def _mfcc_delta_features(audio, sr, n_mfcc=N_MFCC):
    """
    MFCC Delta coefficients (first-order temporal derivative).
    Motivation: Delta-MFCC captures *rate of change* of the spectral envelope,
    which encodes prosodic dynamics linked to arousal/valence dimensions.

    Returns: 26 dims  (delta_mean[13] + delta_std[13])
    """
    mfcc       = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    return np.concatenate([np.mean(mfcc_delta, axis=1),
                            np.std(mfcc_delta,  axis=1)])


def _chroma_features(audio, sr, n_chroma=N_CHROMA):
    """
    Chroma STFT — 12-dimensional pitch-class energy distribution.
    Motivation: Emotion differs in harmonic structure; happy speech tends
    toward major tonality while sad leans toward minor intervals.

    Returns: 24 dims  (chroma_mean[12] + chroma_std[12])
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma)
    return np.concatenate([np.mean(chroma, axis=1),
                            np.std(chroma,  axis=1)])


def _spectral_flatness_features(audio):
    """
    Spectral Flatness (Wiener entropy) — ratio of geometric to arithmetic
    mean of the power spectrum.
    Motivation: Angry/excited speech has more tonal energy (low flatness);
    whispered or neutral speech approaches white noise (high flatness).

    Returns: 2 dims  (flatness_mean, flatness_std)
    """
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    return np.array([float(np.mean(flatness)), float(np.std(flatness))])


def _mel_spectrogram_features(audio, sr, n_bands=N_MEL_BANDS):
    """
    Mel-Spectrogram — perceptually weighted frequency representation.
    We compress n_mels=128 filterbanks into N_MEL_BANDS coarser bands by
    uniform pooling, then take mean + std per band.
    Motivation: Mel scale matches human auditory perception; pooled stats
    capture gross spectral shape without high dimensionality.

    Returns: 20 dims  (band_mean[10] + band_std[10])
    """
    mel = librosa.feature.melspectrogram(y=audio, sr=sr,
                                          n_mels=128, power=2.0)
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    # Pool 128 mel bands → n_bands coarser buckets
    band_size = mel_db.shape[0] // n_bands
    means, stds = [], []
    for b in range(n_bands):
        band = mel_db[b * band_size: (b + 1) * band_size, :]
        means.append(float(np.mean(band)))
        stds.append(float(np.std(band)))
    return np.array(means + stds)   # 20


def _tonnetz_features(audio, sr):
    """
    Tonnetz (tonal centroid features) — 6-dimensional harmonic representation
    based on projections onto perfect fifth, minor third, and major third axes.
    Motivation: Tonnetz captures harmonic tension; e.g., surprised speech
    shows sudden tonal shifts that diverge from neutral/sad profiles.

    Returns: 12 dims  (tonnetz_mean[6] + tonnetz_std[6])
    """
    # Tonnetz requires harmonic component; use harmonic separation
    audio_harm = librosa.effects.harmonic(audio)
    tonnetz    = librosa.feature.tonnetz(y=audio_harm, sr=sr)   # (6, T)
    return np.concatenate([np.mean(tonnetz, axis=1),
                            np.std(tonnetz,  axis=1)])


# ── Phase 2 Master Extractor ──────────────────────────────────────────────────

def extract_all_features_phase2(file_path, sr=SAMPLE_RATE):
    """
    Phase 2 feature pipeline (135 dims).

    Composition:
      [MFCC       26] + [STE/ZCR    3] + [F0         2] + [Spectral   20]
      [MFCC-Delta 26] + [Chroma    24] + [Flatness   2] + [Mel-Spec  20]
      [Tonnetz    12]
      ─────────────────────────────────────────────────────────────────────
      TOTAL = 51 (Phase 1) + 84 (Phase 2) = 135 dims
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as exc:
        print(f"  [WARN] Cannot load '{os.path.basename(file_path)}': {exc}")
        return None

    audio = preprocess_audio(audio, sr)

    fl = _ms_to_samples(FRAME_DURATION_MS, sr)
    hl = _ms_to_samples(HOP_DURATION_MS, sr)

    if len(audio) < fl:
        print(f"  [WARN] Too short: {os.path.basename(file_path)}")
        return None

    # ── Phase 1 features ────────────────────────────────────────────────────
    mfcc_feats     = _mfcc_features(audio, sr)
    ste_m, zcr_m, zcr_s = _ste_zcr(audio, fl, hl)
    f0_m, f0_s     = _pitch_features(audio, sr)
    spec_feats     = _spectral_features(audio, sr)

    # ── Phase 2 new features ─────────────────────────────────────────────────
    delta_feats    = _mfcc_delta_features(audio, sr)
    chroma_feats   = _chroma_features(audio, sr)
    flatness_feats = _spectral_flatness_features(audio)
    mel_feats      = _mel_spectrogram_features(audio, sr)
    tonnetz_feats  = _tonnetz_features(audio, sr)

    return np.concatenate([
        mfcc_feats,                        # 26
        [ste_m, zcr_m, zcr_s],            # 3
        [f0_m, f0_s],                      # 2
        spec_feats,                        # 20
        delta_feats,                       # 26  ← NEW
        chroma_feats,                      # 24  ← NEW
        flatness_feats,                    # 2   ← NEW
        mel_feats,                         # 20  ← NEW
        tonnetz_feats,                     # 12  ← NEW
    ])                                     # total = 135


def get_feature_names_phase2(n_mfcc=N_MFCC):
    """Human-readable feature names (same order as extract_all_features_phase2)."""
    names = []
    # Phase 1
    names += [f"MFCC_{i+1}_mean" for i in range(n_mfcc)]
    names += [f"MFCC_{i+1}_std"  for i in range(n_mfcc)]
    names += ["STE_mean", "ZCR_mean", "ZCR_std"]
    names += ["F0_mean", "F0_std"]
    names += ["Centroid_mean", "Centroid_std",
              "Bandwidth_mean", "Bandwidth_std",
              "Rolloff_mean",   "Rolloff_std"]
    names += [f"Contrast_b{i+1}_mean" for i in range(7)]
    names += [f"Contrast_b{i+1}_std"  for i in range(7)]
    # Phase 2
    names += [f"dMFCC_{i+1}_mean" for i in range(n_mfcc)]
    names += [f"dMFCC_{i+1}_std"  for i in range(n_mfcc)]
    names += [f"Chroma_{i+1}_mean" for i in range(12)]
    names += [f"Chroma_{i+1}_std"  for i in range(12)]
    names += ["Flatness_mean", "Flatness_std"]
    names += [f"Mel_band{i+1}_mean" for i in range(N_MEL_BANDS)]
    names += [f"Mel_band{i+1}_std"  for i in range(N_MEL_BANDS)]
    names += [f"Tonnetz_{i+1}_mean" for i in range(6)]
    names += [f"Tonnetz_{i+1}_std"  for i in range(6)]
    return names   # 135 total


# ===========================================================================
# SECTION 4 — FEATURE MATRIX
# ===========================================================================

def build_feature_matrix(file_paths, labels, sr=SAMPLE_RATE):
    X_list, y_list, valid_paths = [], [], []
    total = len(file_paths)
    for idx, (fp, label) in enumerate(zip(file_paths, labels)):
        print(f"  [{idx+1}/{total}] {os.path.basename(fp)} ...",
              end=" ", flush=True)
        feats = extract_all_features_phase2(fp, sr=sr)
        if feats is not None:
            X_list.append(feats)
            y_list.append(label)
            valid_paths.append(fp)
            print("OK")
        else:
            print("SKIP")
    X = np.array(X_list)
    print(f"\n[Feature Matrix] Shape: {X.shape} | "
          f"Classes: {sorted(set(y_list))}")
    return X, y_list, valid_paths


# ===========================================================================
# SECTION 5 — MODEL DEFINITIONS
# ===========================================================================

def build_rf(random_state=RANDOM_STATE):
    """
    Phase 2 RandomForest — n_estimators raised to 300.
    Serves as the stable ensemble anchor and feature-importance reference.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgb(random_state=RANDOM_STATE):
    """
    XGBoostClassifier — gradient boosting on decision trees.
    Advantages over RF: sequential boosting corrects prior errors;
    built-in L1/L2 regularisation; handles feature collinearity better.
    """
    if not XGBOOST_AVAILABLE:
        return None
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def build_mlp(random_state=RANDOM_STATE):
    """
    MLPClassifier (Multi-Layer Perceptron).
    Architecture: 256 → 128 → 64 neurons, ReLU activation, dropout-style
    early stopping.
    Motivation: Non-linear interactions between acoustic features (e.g.,
    high pitch + high energy = anger) benefit from learned hidden
    representations.
    """
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,               # L2 regularisation
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )


# ===========================================================================
# SECTION 6 — HYPERPARAMETER OPTIMISATION  (RandomizedSearchCV)
# ===========================================================================

def tune_rf(X_train, y_train, n_iter=20, cv=3, random_state=RANDOM_STATE):
    """
    RandomizedSearchCV for RandomForest.
    n_iter=20 gives ~95% probability of landing within top 5% of the
    parameter space at a fraction of GridSearch cost.
    """
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth":    [10, 15, 20, None],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
    }
    base = RandomForestClassifier(
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"  [RF HPO] Best CV acc : {search.best_score_*100:.1f}%  "
          f"({time.time()-t0:.0f}s)")
    print(f"  [RF HPO] Best params : {search.best_params_}")
    return search.best_estimator_


def tune_xgb(X_train, y_train, le,
             n_iter=20, cv=3, random_state=RANDOM_STATE):
    """RandomizedSearchCV for XGBoost."""
    if not XGBOOST_AVAILABLE:
        return None
    param_dist = {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 5, 6, 8, 10],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.7, 0.8, 1.0],
        "reg_alpha":       [0, 0.1, 0.5, 1.0],
        "reg_lambda":      [1.0, 1.5, 2.0],
    }
    y_enc = le.transform(y_train)
    base  = XGBClassifier(
        eval_metric="mlogloss",
        random_state=random_state, n_jobs=-1, verbosity=0,
    )
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    t0 = time.time()
    search.fit(X_train, y_enc)
    print(f"  [XGB HPO] Best CV acc : {search.best_score_*100:.1f}%  "
          f"({time.time()-t0:.0f}s)")
    print(f"  [XGB HPO] Best params : {search.best_params_}")
    return search.best_estimator_


def tune_mlp(X_train, y_train,
             n_iter=15, cv=3, random_state=RANDOM_STATE):
    """RandomizedSearchCV for MLP."""
    param_dist = {
        "hidden_layer_sizes": [
            (128,), (256,), (128, 64), (256, 128),
            (256, 128, 64), (512, 256, 128),
        ],
        "alpha":            [1e-4, 1e-3, 1e-2, 1e-1],
        "learning_rate_init":[1e-4, 5e-4, 1e-3, 5e-3],
    }
    base = MLPClassifier(
        activation="relu", solver="adam",
        max_iter=500, early_stopping=False,
        validation_fraction=0.1, n_iter_no_change=20,
        random_state=random_state,
    )
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="accuracy",
        n_jobs=1,              # MLP is not thread-safe inside n_jobs
        random_state=random_state,
        verbose=1,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"  [MLP HPO] Best CV acc : {search.best_score_*100:.1f}%  "
          f"({time.time()-t0:.0f}s)")
    print(f"  [MLP HPO] Best params : {search.best_params_}")
    return search.best_estimator_


# ===========================================================================
# SECTION 7 — EVALUATION UTILITIES
# ===========================================================================

def evaluate_model(name, clf, X_test, y_test, class_names,
                   cm_path=None):
    """Accuracy + classification report + confusion matrix PNG."""
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print(f"  {name.upper()} — RESULTS")
    print("=" * 65)
    print(classification_report(y_test, y_pred,
                                 target_names=class_names, zero_division=0))
    print(f"  Overall Accuracy : {accuracy*100:.1f}%")
    print("=" * 65)

    if cm_path:
        cm  = confusion_matrix(y_test, y_pred, labels=class_names)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=class_names).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"Confusion Matrix — {name}\nAccuracy: {accuracy*100:.1f}%",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Plot] Saved → {cm_path}")

    return accuracy, y_pred


def cross_validate(name, clf, X_scaled, y, cv=5):
    """StratifiedKFold cross-validation."""
    skf    = StratifiedKFold(n_splits=cv, shuffle=True,
                              random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_scaled, y, cv=skf,
                              scoring="accuracy", n_jobs=-1)
    print(f"\n[Cross-Val — {name}] {cv}-Fold: "
          f"{scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print(f"  Per-fold: {[f'{s*100:.1f}%' for s in scores]}")
    return scores


# ===========================================================================
# SECTION 8 — ERROR ANALYSIS  (Phase 3 guidance)
# ===========================================================================

def error_analysis(ensemble_clf, X_test, y_test, y_pred,
                   class_names, save_prefix="group9_phase2"):
    """
    1. Per-class precision/recall heatmap
    2. Confusion pair ranking  (which classes are most confused?)
    3. Printed Phase 3 strategy recommendations
    """
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    # ── 1. Normalised confusion heatmap ─────────────────────────────────────
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="YlOrRd", linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("Normalised Confusion Matrix — Phase 2 Ensemble", fontsize=12)
    plt.tight_layout()
    path_hm = f"{save_prefix}_heatmap.png"
    plt.savefig(path_hm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Error Analysis] Heatmap saved → {path_hm}")

    # ── 2. Confusion pair ranking ────────────────────────────────────────────
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true":       class_names[i],
                    "predicted":  class_names[j],
                    "count":      int(cm[i, j]),
                    "rate":       float(cm_norm[i, j]),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)

    print("\n[Error Analysis] Top confusion pairs (true → predicted):")
    print(f"  {'True':<12} {'Predicted':<12} {'Count':>6}  {'Rate':>6}")
    print(f"  {'-'*12} {'-'*12} {'-'*6}  {'-'*6}")
    for p in pairs[:10]:
        print(f"  {p['true']:<12} {p['predicted']:<12} "
              f"{p['count']:>6}  {p['rate']:>5.1%}")

    # ── 3. Phase 3 strategy recommendations ─────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 3 STRATEGY RECOMMENDATIONS (based on error analysis)")
    print("=" * 65)
    if pairs:
        top_pair = pairs[0]
        print(f"\n  Most confused pair: "
              f"'{top_pair['true']}' → '{top_pair['predicted']}' "
              f"({top_pair['rate']:.1%} misclassification rate)")
        print("""
  Suggested mitigation strategies:
  ─────────────────────────────────
  A. Feature Engineering:
     • Add MFCC Delta-Delta (2nd order) → captures acceleration of
       spectral envelope; useful for sad↔neutral which differ in
       speech rate deceleration.
     • Add voiced/unvoiced ratio → ratio of voiced frames (F0>0)
       to total; angry speech has high voiced ratio.
     • Add speaking rate (syllable nuclei per second) via onset
       strength peaks.

  B. Class-Specific Thresholding:
     • After probability calibration (CalibratedClassifierCV),
       lower the decision threshold for under-recalled classes.
     • Example: if 'sad' recall < 0.5, set threshold_sad = 0.35.

  C. Augmentation for Hard Classes:
     • Time-stretch confused classes (±10%) to synthesise variants.
     • Pitch-shift ±1–2 semitones; adds intra-class variance.

  D. Cost-Sensitive Learning:
     • Increase misclassification cost for the confused pair via
       sample_weight in XGBClassifier / class_weight in RF.

  E. Hierarchical Classification:
     • Stage 1: binary arousal split (high: angry/happy/surprised
       vs low: neutral/sad).
     • Stage 2: fine-grained within each arousal group.
       This exploits the known 2D valence-arousal emotion space.
""")

    return pairs


# ===========================================================================
# SECTION 9 — FEATURE IMPORTANCE (Top-N bar chart)
# ===========================================================================

def plot_feature_importance(rf_clf, n_features_total,
                             save_path="group9_phase2_feature_importance.png",
                             top_n=30):
    names       = get_feature_names_phase2()[:n_features_total]
    importances = rf_clf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(top_n), importances[indices],
           color="#2E75B6", edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([names[i] for i in indices],
                        rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance Score")
    ax.set_title(f"Top-{top_n} Feature Importances — Phase 2 RF (135-dim)",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Feature importance saved → {save_path}")


# ===========================================================================
# SECTION 10 — SAVE RESULTS
# ===========================================================================

def save_results_csv(file_paths, y_true, y_pred,
                     model_scores, output_csv=OUTPUT_CSV):
    df_pred = pd.DataFrame({
        "file_name":  [os.path.basename(p) for p in file_paths],
        "true_label": list(y_true),
        "predicted":  list(y_pred),
        "correct":    [t == p for t, p in zip(y_true, y_pred)],
    })
    df_pred.to_csv(output_csv, index=False)
    print(f"[Results] Predictions saved → {output_csv}")

    summary_path = output_csv.replace(".csv", "_model_summary.csv")
    df_summary = pd.DataFrame(model_scores)
    df_summary.to_csv(summary_path, index=False)
    print(f"[Results] Model summary saved → {summary_path}")


# ===========================================================================
# SECTION 11 — MAIN ENTRY POINT
# ===========================================================================

def main():
    print("=" * 65)
    print("  COE216 Final Project — Emo Challenge 2026")
    print("  Phase 2: Research & Development")
    print("  Group 9 — 135-dim | RF + XGB + MLP + Ensemble")
    print("=" * 65 + "\n")

    # ── Step 1: Load dataset ─────────────────────────────────────────────────
    print("[Step 1] Loading dataset ...")
    file_paths, labels = load_dataset()

    # ── Step 2: Extract Phase 2 features ────────────────────────────────────
    print("[Step 2] Extracting Phase 2 feature set (135 dims) ...")
    X, y, valid_paths = build_feature_matrix(file_paths, labels)

    if len(X) == 0:
        print("[ERROR] No features extracted. Check DATASET_ROOT.")
        return

    print(f"\n[INFO] Feature vector: {X.shape[1]} dims | "
          f"Samples: {X.shape[0]}")

    # ── Step 3: Encode labels ─────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(EMOTION_CLASSES)
    y_enc       = le.transform(y)
    class_names = list(le.classes_)

    # ── Step 4: Scale features ─────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n[Scaler] Applied. Range: "
          f"[{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

    # ── Step 5: Train/test split ───────────────────────────────────────────────
    (X_train, X_test,
     y_train, y_test,
     paths_train, paths_test) = train_test_split(
        X_scaled, y, valid_paths,
        test_size=0.20, random_state=RANDOM_STATE, stratify=y_enc,
    )
    print(f"\n[Split] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Step 6: Hyperparameter optimisation ───────────────────────────────────
    print("\n[Step 6a] Tuning RandomForest ...")
    rf_tuned = tune_rf(X_train, y_train)

    print("\n[Step 6b] Tuning XGBoost ...")
    xgb_tuned = tune_xgb(X_train, y_train, le)

    print("\n[Step 6c] Tuning MLP ...")
    mlp_tuned = tune_mlp(X_train, y_train)

    # ── Step 7: Train tuned models ─────────────────────────────────────────────
    print("\n[Step 7] Training tuned models on full training set ...")
    rf_tuned.fit(X_train, y_train)

    if xgb_tuned is not None:
        xgb_tuned.fit(X_train, le.transform(y_train))

    mlp_tuned.fit(X_train, y_train)

    # ── Step 8: Build Soft-Voting Ensemble ────────────────────────────────────
    print("\n[Step 8] Building Soft-Voting Ensemble ...")
    estimators = [("rf", rf_tuned), ("mlp", mlp_tuned)]
    if xgb_tuned is not None:
        # Wrap XGB (integer labels) so it works inside VotingClassifier (string labels)
        from sklearn.base import BaseEstimator, ClassifierMixin
        class XGBWrapper(BaseEstimator, ClassifierMixin):
            """Sklearn-compatible adapter: converts string labels ↔ integer indices."""
            def __init__(self, xgb_clf, label_encoder):
                self.xgb_clf = xgb_clf
                self.label_encoder = label_encoder

            def fit(self, X, y):
                self.classes_ = np.array(self.label_encoder.classes_)
                self.xgb_clf.fit(X, self.label_encoder.transform(y))
                return self

            def predict(self, X):
                return self.label_encoder.inverse_transform(self.xgb_clf.predict(X))

            def predict_proba(self, X):
                return self.xgb_clf.predict_proba(X)

            def get_params(self, deep=True):
                return {"xgb_clf": self.xgb_clf, "label_encoder": self.label_encoder}

        xgb_wrapped = XGBWrapper(xgb_tuned, le)
        xgb_wrapped.fit(X_train, y_train)  # sets classes_
        estimators.append(("xgb", xgb_wrapped))

    # Note: VotingClassifier with soft voting averages predict_proba outputs
    # n_jobs=1 to avoid pickling issues with inner wrapper class
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
    ensemble.fit(X_train, y_train)
    print("  Ensemble trained.")

    # ── Step 9: Evaluate all models ───────────────────────────────────────────
    print("\n[Step 9] Evaluating models on test set ...")
    model_scores = []

    acc_rf, pred_rf = evaluate_model(
        "RF (Phase 2 Tuned)", rf_tuned,
        X_test, y_test, class_names,
        cm_path="group9_phase2_cm_rf.png",
    )
    model_scores.append({"model": "RF_tuned", "test_accuracy": acc_rf})

    if xgb_tuned is not None:
        y_test_enc  = le.transform(y_test)
        y_pred_xgb  = le.inverse_transform(xgb_tuned.predict(X_test))
        acc_xgb     = accuracy_score(y_test, y_pred_xgb)
        print(f"\n  XGBoost Accuracy : {acc_xgb*100:.1f}%")
        model_scores.append({"model": "XGB_tuned", "test_accuracy": acc_xgb})

    acc_mlp, pred_mlp = evaluate_model(
        "MLP (Tuned)", mlp_tuned,
        X_test, y_test, class_names,
        cm_path="group9_phase2_cm_mlp.png",
    )
    model_scores.append({"model": "MLP_tuned", "test_accuracy": acc_mlp})

    acc_ens, pred_ens = evaluate_model(
        "Soft-Voting Ensemble", ensemble,
        X_test, y_test, class_names,
        cm_path="group9_phase2_cm_ensemble.png",
    )
    model_scores.append({"model": "Ensemble", "test_accuracy": acc_ens})

    # ── Step 10: Cross-validation on ensemble ────────────────────────────────
    print("\n[Step 10] 5-fold Cross-validation (Ensemble) ...")
    cv_scores = cross_validate("Ensemble", ensemble, X_scaled, y)

    # ── Step 11: Feature importance (RF) ─────────────────────────────────────
    print("\n[Step 11] Feature importance plot ...")
    plot_feature_importance(rf_tuned, X.shape[1])

    # ── Step 12: Error analysis ───────────────────────────────────────────────
    print("\n[Step 12] Error analysis ...")
    error_analysis(ensemble, X_test, y_test, pred_ens,
                   class_names, save_prefix="group9_phase2")

    # ── Step 13: Save results ─────────────────────────────────────────────────
    print("\n[Step 13] Saving results ...")
    save_results_csv(paths_test, y_test, pred_ens, model_scores)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 2 COMPLETE — Model Comparison")
    print("=" * 65)
    print(f"  Feature dims    : {X.shape[1]}  (51 Phase1 + 84 Phase2)")
    print(f"  Train samples   : {len(X_train)}")
    print(f"  Test  samples   : {len(X_test)}")
    print()
    for ms in model_scores:
        bar = "█" * int(ms["test_accuracy"] * 30)
        print(f"  {ms['model']:<20} {ms['test_accuracy']*100:5.1f}%  {bar}")
    print()
    print(f"  Ensemble CV     : {cv_scores.mean()*100:.1f}% "
          f"± {cv_scores.std()*100:.1f}%")
    print("=" * 65)
    print("\n  Output files:")
    print(f"    • {OUTPUT_CSV}")
    print( "    • group9_phase2_cm_rf.png")
    print( "    • group9_phase2_cm_mlp.png")
    print( "    • group9_phase2_cm_ensemble.png")
    print( "    • group9_phase2_heatmap.png")
    print( "    • group9_phase2_feature_importance.png")
    print("\n  → Leaderboard:")
    print("     https://bil216finalproje-woau7utnbhu7q6hbz8nuff.streamlit.app/")


if __name__ == "__main__":
    main()
