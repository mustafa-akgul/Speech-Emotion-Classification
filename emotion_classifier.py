# -*- coding: utf-8 -*-
"""
=============================================================================
Speech Emotion Classification
Random Forest | 51-dim Feature Set | 5-class

Authors: Ahmet Akin, Berivan Demir, Mustafa Talha Akgul

Description:
    Speech emotion classification using a 51-dimensional feature
    set and a RandomForestClassifier with StandardScaler.

    FEATURE ENGINEERING (51 dims):
      - 13 MFCC mean + std                       (26 dims)
      - Spectral Centroid, Bandwidth, Rolloff     ( 6 dims)
      - Spectral Contrast (7 bands) mean + std   (14 dims)
      - STE mean, ZCR mean + std                  ( 3 dims)
      - Pitch (F0) mean + std                     ( 2 dims)
      Total: 26 + 6 + 14 + 3 + 2 = 51 dims

    PREPROCESSING:
      - librosa.effects.preemphasis  (signal clarity)
      - librosa.effects.trim         (silence removal)

    MODEL:
      - StandardScaler  (feature normalization)
      - RandomForestClassifier (n_estimators=200, max_depth=15,
                                class_weight='balanced')
      - StratifiedKFold (5-fold) cross-validation

Dataset Layout (must match this structure):
    Midterm_Dataset_2026/
        <group_folder>/
            *.wav

NOTE: All variable names, paths, and function names use ASCII-only
      characters to prevent UnicodeDecodeError on any OS.
=============================================================================
"""

# ─────────────────────────── Standard Imports ───────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")             # Headless backend — no display required
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")


# ===========================================================================
# SECTION 1 — CONFIGURATION
# ===========================================================================

DATASET_ROOT    = "Midterm_Dataset_2026"
EMOTION_CLASSES = ["neutral", "happy", "angry", "sad", "surprised"]

EMOTION_MAPPING = {
    # Türkçe — ASCII
    "notr":      "neutral",
    "mutlu":     "happy",
    "ofkeli":    "angry",
    "uzgun":     "sad",
    "saskin":    "surprised",
    "sasirma":   "surprised",
    # Türkçe — özel karakter (ö, ü, ş)
    "nötr":      "neutral",
    "öfkeli":    "angry",
    "üzgün":     "sad",
    "şaşkın":    "surprised",
    "şaşırma":   "surprised",
    # İngilizce
    "neutral":   "neutral",
    "happy":     "happy",
    "angry":     "angry",
    "furious":   "angry",
    "sad":       "sad",
    "surprised": "surprised",
    "shocked":   "surprised",
}

SAMPLE_RATE       = 22050
FRAME_DURATION_MS = 25
HOP_DURATION_MS   = 10

N_MFCC = 13

# ── Model Hyper-parameters ───────────────────────────────────────────────────
N_ESTIMATORS = 200
MAX_DEPTH    = 15
RANDOM_STATE = 42

OUTPUT_CSV = "results.csv"
OUTPUT_CM  = "confusion_matrix.png"


# ===========================================================================
# SECTION 2 — DATASET LOADING
# ===========================================================================

def load_dataset(dataset_root=DATASET_ROOT, emotion_classes=EMOTION_CLASSES):
    """
    Scan the dataset folder tree and return a list of (filepath, label) pairs.
    """
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset root folder not found: '{dataset_root}'\n"
            "Please set DATASET_ROOT to the correct path."
        )

    file_paths, labels = [], []

    for group_folder in os.listdir(dataset_root):
        group_path = os.path.join(dataset_root, group_folder)
        if not os.path.isdir(group_path):
            continue
        print(f"  [Dataset] Scanning group: {group_folder}")

        for filename in os.listdir(group_path):
            if not filename.lower().endswith(".wav"):
                continue
            emotion = extract_emotion_from_filename(filename)
            if emotion and emotion in emotion_classes:
                file_paths.append(os.path.join(group_path, filename))
                labels.append(emotion)
            else:
                print(f"    [SKIP] Unknown emotion in: {filename}")

    print(f"\n[Dataset] Total: {len(file_paths)} files across "
          f"{len(set(labels))} classes.\n")
    return file_paths, labels


def extract_emotion_from_filename(filename):
    """Extract emotion label from filename like G04_D01_E_21_Mutlu_C2.wav"""
    name = os.path.splitext(filename)[0]   # uzantıyı at
    parts = name.lower().split("_")
    for part in parts:
        part_clean = part.strip().rstrip(".")  # nokta/boşluk temizle
        if part_clean in EMOTION_MAPPING:
            return EMOTION_MAPPING[part_clean]
    return None


# ===========================================================================
# SECTION 3 — FEATURE EXTRACTION  (Phase 1 — 51 dims)
# ===========================================================================

def ms_to_samples(ms, sr):
    return int(sr * ms / 1000)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_audio(audio, sr=SAMPLE_RATE):
    """
    Apply preemphasis filtering and trim leading/trailing silence.

    Steps:
      1. preemphasis  — amplifies high-frequency components for clearer
                        consonant/fricative detail (coeff=0.97 standard)
      2. trim         — removes silence regions below the threshold
    """
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    return audio


# ── Time-domain ───────────────────────────────────────────────────────────────

def extract_ste_and_zcr(audio, frame_length, hop_length):
    """
    Short-Time Energy (mean) + Zero Crossing Rate (mean + std).

    Returns: (ste_mean, zcr_mean, zcr_std)  →  3 dims
    """
    frames = librosa.util.frame(audio, frame_length=frame_length,
                                 hop_length=hop_length)
    ste = np.sum(frames ** 2, axis=0)
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    return float(np.mean(ste)), float(np.mean(zcr)), float(np.std(zcr))


def extract_pitch_features(audio, sr):
    """
    Fundamental frequency (F0) estimated via autocorrelation.

    Returns: (f0_mean, f0_std)  →  2 dims
    """
    frame_len = ms_to_samples(FRAME_DURATION_MS, sr)
    hop_len   = ms_to_samples(HOP_DURATION_MS,   sr)
    frames    = librosa.util.frame(audio, frame_length=frame_len,
                                    hop_length=hop_len)
    f0_values = []
    lag_min   = max(1, int(sr / 500))
    lag_max   = int(sr / 50)

    for i in range(frames.shape[1]):
        frame = frames[:, i]
        r     = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if lag_max >= len(r) or r[0] < 1e-12:
            continue
        best_lag   = int(np.argmax(r[lag_min: lag_max + 1])) + lag_min
        confidence = r[best_lag] / r[0]
        if confidence >= 0.3:
            f0_values.append(float(sr) / float(best_lag))

    if f0_values:
        return float(np.mean(f0_values)), float(np.std(f0_values))
    return 0.0, 0.0


# ── Spectral ──────────────────────────────────────────────────────────────────

def extract_spectral_features(audio, sr):
    """
    Spectral Centroid, Bandwidth, Rolloff (mean + std) + Spectral Contrast
    (7 bands, mean + std).

    Returns: 6 + 14 = 20 dims
    """
    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    contrast  = librosa.feature.spectral_contrast(y=audio, sr=sr)   # (7, T)

    feats = np.array([
        np.mean(centroid),  np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff),   np.std(rolloff),
    ])
    feats = np.concatenate([feats,
                             np.mean(contrast, axis=1),
                             np.std(contrast,  axis=1)])
    return feats   # 20 dims


# ── MFCC ──────────────────────────────────────────────────────────────────────

def extract_mfcc_features(audio, sr, n_mfcc=N_MFCC):
    """
    13 MFCC mean + 13 MFCC std.

    Returns: 2 × n_mfcc = 26 dims
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([
        np.mean(mfcc, axis=1),   # 13 dims
        np.std(mfcc,  axis=1),   # 13 dims
    ])


# ── Feature Name Labels ───────────────────────────────────────────────────────

def get_feature_names(n_mfcc=N_MFCC):
    """Return human-readable feature names (same order as extract_all_features)."""
    names = []

    # MFCC block (2 × n_mfcc = 26)
    names += [f"MFCC_{i+1}_mean" for i in range(n_mfcc)]
    names += [f"MFCC_{i+1}_std"  for i in range(n_mfcc)]

    # Time-domain (3)
    names += ["STE_mean", "ZCR_mean", "ZCR_std"]

    # Pitch (2)
    names += ["F0_mean", "F0_std"]

    # Spectral (20)
    names += ["Centroid_mean", "Centroid_std",
              "Bandwidth_mean", "Bandwidth_std",
              "Rolloff_mean",   "Rolloff_std"]
    names += [f"Contrast_b{i+1}_mean" for i in range(7)]
    names += [f"Contrast_b{i+1}_std"  for i in range(7)]

    return names   # 51 total


# ── Master Feature Extractor ──────────────────────────────────────────────────

def extract_all_features(file_path, sr=SAMPLE_RATE):
    """
    Phase 1 feature pipeline (51 dims).

      [MFCC 26] + [STE/ZCR 3] + [F0 2] + [Spectral 20] = 51

    Preprocessing applied before extraction:
      - preemphasis  (coef=0.97)
      - trim silence (top_db=20)
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as exc:
        print(f"  [WARN] Could not load '{os.path.basename(file_path)}': {exc}")
        return None

    # ── Preprocessing ────────────────────────────────────────────────────────
    audio = preprocess_audio(audio, sr=sr)

    if len(audio) < ms_to_samples(FRAME_DURATION_MS, sr):
        print(f"  [WARN] Audio too short after trim: {os.path.basename(file_path)}")
        return None

    frame_len = ms_to_samples(FRAME_DURATION_MS, sr)
    hop_len   = ms_to_samples(HOP_DURATION_MS,   sr)

    # ── Time-domain ──────────────────────────────────────────────────────────
    ste_mean, zcr_mean, zcr_std = extract_ste_and_zcr(audio, frame_len, hop_len)
    f0_mean,  f0_std            = extract_pitch_features(audio, sr)

    # ── Frequency-domain ─────────────────────────────────────────────────────
    mfcc_feats     = extract_mfcc_features(audio, sr)
    spectral_feats = extract_spectral_features(audio, sr)

    feature_vector = np.concatenate([
        mfcc_feats,                        # 26 dims
        [ste_mean, zcr_mean, zcr_std],     # 3 dims
        [f0_mean,  f0_std],                # 2 dims
        spectral_feats,                    # 20 dims
    ])                                     # total = 51 dims

    return feature_vector


# ===========================================================================
# SECTION 4 — FEATURE MATRIX
# ===========================================================================

def build_feature_matrix(file_paths, labels, sr=SAMPLE_RATE):
    X_list, y_list, valid_paths = [], [], []
    total = len(file_paths)

    for idx, (fp, label) in enumerate(zip(file_paths, labels)):
        print(f"  [{idx + 1}/{total}] Extracting: {os.path.basename(fp)} ...",
              end=" ", flush=True)
        feats = extract_all_features(fp, sr=sr)
        if feats is not None:
            X_list.append(feats)
            y_list.append(label)
            valid_paths.append(fp)
            print("OK")
        else:
            print("SKIP")

    X = np.array(X_list)
    print(f"\n[Feature Matrix] Shape: {X.shape} | Classes: {sorted(set(y_list))}")
    return X, y_list, valid_paths


# ===========================================================================
# SECTION 5 — MODEL TRAINING AND EVALUATION
# ===========================================================================

def build_model(random_state=RANDOM_STATE):
    """
    Phase 1 single-model pipeline.

    Scaler  : StandardScaler — zero mean, unit variance per feature
    Classifier: RandomForestClassifier
        n_estimators  = 200    (enough trees for stable votes)
        max_depth     = 15     (limits overfitting while preserving expressiveness)
        class_weight  = 'balanced'  (handles class imbalance automatically)
    """
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return rf


def evaluate_model(clf, X_test, y_test, class_names,
                   model_label="Phase 1 RF", output_cm=OUTPUT_CM):
    """Evaluate classifier and save confusion matrix PNG."""
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print(f"  {model_label.upper()} — CLASSIFICATION RESULTS")
    print("=" * 65)
    print(classification_report(y_test, y_pred,
                                 target_names=class_names,
                                 zero_division=0))
    print(f"  Overall Accuracy : {accuracy * 100:.1f}%")
    print("=" * 65)

    cm  = confusion_matrix(y_test, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=class_names).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(
        f"Confusion Matrix — {model_label}\n"
        f"Overall Accuracy: {accuracy * 100:.1f}%",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(output_cm, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Confusion matrix saved -> {output_cm}")
    plt.close()

    return accuracy, y_pred


def cross_validate_model(clf, X_scaled, y, cv=5, label="RF Phase 1"):
    """
    StratifiedKFold cross-validation on already-scaled features.

    Note: scaler is fit on the full feature matrix before this call;
    for a production pipeline wrap scaler + clf in sklearn.pipeline.Pipeline.
    """
    skf       = StratifiedKFold(n_splits=cv, shuffle=True,
                                 random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=skf,
                                 scoring="accuracy", n_jobs=-1)
    print(f"\n[Cross-Val — {label}] {cv}-Fold: "
          f"{cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%")
    print(f"  Per-fold: {[f'{s*100:.1f}%' for s in cv_scores]}")
    return cv_scores


def plot_feature_importance(clf_rf, n_features_total,
                             save_path="feature_importance.png",
                             top_n=25):
    """Bar chart of top-N most important features."""
    names       = get_feature_names()[:n_features_total]
    importances = clf_rf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(top_n), importances[indices],
           color="#2E75B6", edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([names[i] for i in indices],
                        rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance Score")
    ax.set_title(f"Top-{top_n} Feature Importances — Phase 1 RF", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Feature importance saved -> {save_path}")
    plt.close()


# ===========================================================================
# SECTION 6 — SAVE RESULTS
# ===========================================================================

def save_results_csv(file_paths, y_true, y_pred, output_csv=OUTPUT_CSV):
    df = pd.DataFrame({
        "file_name":  [os.path.basename(p) for p in file_paths],
        "true_label": list(y_true),
        "predicted":  list(y_pred),
        "correct":    [t == p for t, p in zip(y_true, y_pred)],
    })
    df.to_csv(output_csv, index=False)
    print(f"[Results] Saved -> {output_csv}")


# ===========================================================================
# SECTION 7 — MAIN ENTRY POINT
# ===========================================================================

def main():
    print("=" * 65)
    print("  Speech Emotion Classification")
    print("  RandomForest + StandardScaler + 51-dim Features")
    print("=" * 65 + "\n")

    # ── Step 1: Load dataset ─────────────────────────────────────────────────
    print("[Step 1] Loading dataset ...")
    file_paths, labels = load_dataset()

    # ── Step 2: Extract features ─────────────────────────────────────────────
    print("[Step 2] Extracting Phase 1 feature set (51 dims) ...")
    X, y, valid_paths = build_feature_matrix(file_paths, labels)

    if len(X) == 0:
        print("[ERROR] No features extracted. Check DATASET_ROOT.")
        return

    print(f"\n[INFO] Feature vector size: {X.shape[1]} dims")

    # ── Step 3: Encode labels ─────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(EMOTION_CLASSES)
    y_enc       = le.transform(y)
    class_names = list(le.classes_)

    # ── Step 4: Normalize features with StandardScaler ────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n[Scaler] StandardScaler applied. Feature range: "
          f"[{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

    # ── Step 5: Train / test split ────────────────────────────────────────────
    (X_train, X_test,
     y_train, y_test,
     paths_train, paths_test) = train_test_split(
        X_scaled, y_enc, valid_paths,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )
    y_train_str = le.inverse_transform(y_train)
    y_test_str  = le.inverse_transform(y_test)
    print(f"\n[Split] Train: {len(X_train)} | Test: {len(X_test)} samples")

    # ── Step 6: Train Phase 1 RF ──────────────────────────────────────────────
    print("\n[Step 6] Training Phase 1 RandomForestClassifier ...")
    clf = build_model()
    clf.fit(X_train, y_train_str)
    print("  Training complete.")

    # ── Step 7: Evaluate on test set ─────────────────────────────────────────
    print("\n[Step 7] Evaluating on test set ...")
    accuracy, y_pred = evaluate_model(
        clf, X_test, y_test_str, class_names,
        model_label="Phase 1 RF",
        output_cm=OUTPUT_CM,
    )

    # ── Step 8: 5-fold Cross-validation ──────────────────────────────────────
    print("\n[Step 8] StratifiedKFold cross-validation (5-fold) ...")
    cv_scores = cross_validate_model(clf, X_scaled, le.transform(y),
                                      cv=5, label="Phase 1 RF")

    # ── Step 9: Feature importance plot ──────────────────────────────────────
    print("\n[Step 9] Feature importance plot ...")
    plot_feature_importance(clf, X.shape[1])

    # ── Step 10: Save results CSV ─────────────────────────────────────────────
    print("\n[Step 10] Saving results ...")
    save_results_csv(paths_test, y_test_str, y_pred)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DONE")
    print(f"  Feature dims       : {X.shape[1]}")
    print(f"  Preprocessing      : preemphasis + trim")
    print(f"  Scaler             : StandardScaler")
    print(f"  Model              : RandomForest (n={N_ESTIMATORS}, "
          f"depth={MAX_DEPTH}, balanced)")
    print(f"  Test Set Accuracy  : {accuracy * 100:.1f}%")
    print(f"  CV Accuracy (5-fold): "
          f"{cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%")
    print("=" * 65)
    print("\n  Output files:")
    print(f"    * {OUTPUT_CSV}")
    print(f"    * {OUTPUT_CM}")
    print("    * feature_importance.png")


if __name__ == "__main__":
    main()