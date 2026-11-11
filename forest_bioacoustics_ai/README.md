# Forest Bioacoustics AI

A starter research prototype for analysing forest audio recordings and classifying labelled acoustic segments.

## Features
- Upload audio recordings (WAV, MP3, FLAC, OGG, M4A)
- Waveform and mel-spectrogram visualisation
- 5-second segmentation and acoustic feature extraction (MFCC, spectral centroid/bandwidth, zero-crossing rate, RMS)
- Baseline SVM classifier trained from user-provided labelled recordings
- Per-segment candidate predictions and CSV export

## Important scientific limitation
The archive does **not** include a trained species-recognition model or a dataset of labelled animal recordings. It will inspect audio immediately, but species classification is enabled only after you add labelled data and train the baseline model. Model scores are not calibrated probabilities and do not prove species presence. A serious evaluation should use expert-labelled data and independent splits by recording site/date/device, with per-class precision, recall, F1 and a confusion matrix.

## Requirements
- Python 3.10–3.12 recommended
- Internet access for installing Python packages the first time

## Install and run

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```

The app opens in your browser, typically at `http://localhost:8501`.

## Train the baseline classifier
Create one folder per class, with labelled recordings in each:

```text
data/
  train/
    blackbird/
      recording_001.wav
      recording_002.wav
    owl/
      recording_003.wav
      recording_004.wav
```

Then run from the project root:
```bash
python -m app.train
```

The model is written to `models/bioacoustic_model.joblib`. Restart or refresh the Streamlit app.

Use sufficiently varied, correctly labelled clips for every class. Do not put the same source recording or near-identical fragments in both training and evaluation sets. The included feature-based SVM is a baseline, not a state-of-the-art wildlife classifier.

## Suggested next research steps
1. Define the target taxa and geographic region.
2. Collect or obtain recordings with documented licences and verified labels.
3. Establish a site/date/device-aware train-validation-test split.
4. Compare the baseline with a pretrained bioacoustic model (for example, a model designed for bird and soundscape recognition), after checking its licence and label coverage.
5. Report per-class metrics, false positives, uncertainty and out-of-distribution limitations.
6. Treat non-detections cautiously: silence or a low score is not evidence that a species is absent.

## Project layout
```text
app/
  audio.py       audio loading and segmentation
  features.py    acoustic feature extraction
  model.py       model training/persistence/prediction
  train.py       training CLI
  main.py        Streamlit web interface
data/
  samples/       optional example files
tests/           reserved for automated tests
requirements.txt
```

## Troubleshooting
- If package installation fails, use Python 3.10–3.12 and update pip.
- If a file cannot be decoded, convert it to WAV PCM and retry.
- If no model is available, the app still displays waveform and spectrogram.
