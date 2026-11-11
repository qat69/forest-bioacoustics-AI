from pathlib import Path
import numpy as np
import librosa

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

def load_audio(path: str | Path, target_sr: int = 32_000):
    """Load an audio file as mono float32 and resample to target_sr."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(path)
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    if audio.size == 0:
        raise ValueError("Audio file contains no samples.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid numeric values.")
    return audio.astype(np.float32), target_sr

def split_audio(audio: np.ndarray, sr: int, segment_seconds: float = 5.0):
    """Split audio into non-overlapping segments; pad the last segment."""
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    segment_size = int(sr * segment_seconds)
    if segment_size < 1:
        raise ValueError("segment_seconds is too small")
    segments = []
    for start in range(0, len(audio), segment_size):
        chunk = audio[start:start + segment_size]
        if len(chunk) < segment_size:
            chunk = np.pad(chunk, (0, segment_size - len(chunk)))
        segments.append(chunk.astype(np.float32))
    return segments
