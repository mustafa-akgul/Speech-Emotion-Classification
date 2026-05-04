# Speech Emotion Classification

Classifies five emotional states — **Neutral, Happy, Angry, Sad, Surprised** — from short Turkish speech recordings using handcrafted acoustic features and a Random Forest classifier.

## Features

- 51-dimensional feature vector: MFCC (13 coefficients × mean/std), Spectral Centroid / Bandwidth / Rolloff, Spectral Contrast (7 bands), Short-Time Energy, Zero-Crossing Rate, and Pitch (F0)
- Preprocessing: preemphasis filtering + silence trimming
- StandardScaler normalization + Stratified 5-fold cross-validation

## Requirements

```
pip install librosa scikit-learn numpy pandas matplotlib
```

## Usage

Place the dataset under `Midterm_Dataset_2026/` (one subfolder per recording group, WAV files inside), then run:

```bash
python emotion_classifier.py
```

Outputs written to the working directory:
- `group9_phase1_results.csv` — per-file predictions
- `group9_phase1_confusion_matrix.png`
- `group9_phase1_feature_importance.png`

## Dataset Format

Filename convention: `G<group>_D<speaker>_<gender>_<age>_<emotion>_C<quality>.wav`

Supported emotion labels (Turkish and English): `Notr/Nötr`, `Mutlu`, `Öfkeli/Ofkeli`, `Üzgün/Uzgun`, `Şaşkın/Saskin`, `Neutral`, `Happy`, `Angry`, `Sad`, `Surprised`, `Furious`, `Shocked`

## Results (Phase 1)

| Metric | Value |
|---|---|
| Test Accuracy | 62.6% |
| Macro F1 | 0.626 |
| Best class | Sad (F1 = 0.839) |
| Hardest pair | Angry ↔ Happy |
