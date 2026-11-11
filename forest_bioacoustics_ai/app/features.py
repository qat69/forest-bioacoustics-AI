import numpy as np
import librosa

def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a compact feature vector for a baseline classifier.
    This baseline is useful for the prototype; it is not a substitute for
    a model trained and validated on labelled field recordings.
    """
    if audio.size == 0:
        raise ValueError("Audio segment is empty")
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=320, n_mels=64, fmax=min(14_000, sr // 2)
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)
    def stats(x):
        return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])
    return np.concatenate([stats(mfcc), stats(centroid), stats(bandwidth),
                           stats(zcr), stats(rms)]).astype(np.float32)
