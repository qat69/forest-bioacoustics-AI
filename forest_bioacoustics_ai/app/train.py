"""
Train from a labelled folder structure:
data/train/<class_name>/*.wav

Example:
data/train/robin/*.wav
data/train/owl/*.wav

Only train with recordings whose labels and usage rights are known.
For scientific evaluation, split by recording/site/date before training to
avoid leakage between near-duplicate clips.
"""
import sys
from pathlib import Path
import numpy as np

# Allow running as `python app/train.py` or from an IDE:
# make the project root importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio import load_audio, split_audio, SUPPORTED_EXTENSIONS
from app.features import extract_features
from app.model import train_model, save_model

DATA_DIR = Path("data/train")

def main():
    X, y = [], []
    if not DATA_DIR.exists():
        raise SystemExit(f"Training folder not found: {DATA_DIR}")
    class_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    for class_dir in sorted(class_dirs):
        for audio_path in sorted(class_dir.iterdir()):
            if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                audio, sr = load_audio(audio_path)
                for segment in split_audio(audio, sr, segment_seconds=5):
                    X.append(extract_features(segment, sr))
                    y.append(class_dir.name)
            except Exception as exc:
                print(f"Skipping {audio_path}: {exc}")
    if not X:
        raise SystemExit("No usable training audio found in data/train/<class_name>/.")
    model = train_model(np.vstack(X), np.asarray(y))
    save_model(model)
    print(f"Saved model to models/bioacoustic_model.joblib")
    print(f"Segments: {len(y)} | Classes: {len(set(y))}")
    print("Note: these counts are not a substitute for held-out evaluation.")

if __name__ == "__main__":
    main()
