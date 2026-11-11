import sys
from pathlib import Path
import tempfile

# Allow running as `streamlit run app/main.py` or from an IDE:
# make the project root importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt

from app.audio import load_audio, split_audio, SUPPORTED_EXTENSIONS
from app.features import extract_features
from app.model import load_model, predict

st.set_page_config(page_title="Forest Bioacoustics AI", page_icon="🌲", layout="wide")
st.title("🌲 Forest Bioacoustics AI")
st.caption("Prototype for acoustic biodiversity monitoring")

with st.sidebar:
    st.header("About")
    st.write("Upload a field recording to inspect its acoustic features and, if a trained model is available, receive candidate class predictions.")
    st.warning("Predictions depend on training data quality. Probabilities are model scores, not calibrated certainty or proof of species presence.")
    st.write("Supported: WAV, MP3, FLAC, OGG, M4A")

model = load_model()
if model is None:
    st.info("No trained model is installed yet. Audio inspection is available; classification starts after you train a model.")
    st.markdown("**To train:** place labelled audio in `data/train/<class_name>/` and run `python -m app.train`.")
else:
    st.success("Trained model loaded.")

uploaded = st.file_uploader("Choose a forest recording", type=["wav", "mp3", "flac", "ogg", "m4a"])
if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = tmp.name
    try:
        audio, sr = load_audio(temp_path)
        duration = len(audio) / sr
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration", f"{duration:.2f} s")
        c2.metric("Sample rate", f"{sr:,} Hz")
        c3.metric("Peak amplitude", f"{abs(audio).max():.3f}")
        st.audio(uploaded.getvalue(), format=uploaded.type or "audio/wav")

        fig, ax = plt.subplots(figsize=(12, 2.5))
        times = [i / sr for i in range(len(audio))]
        ax.plot(times, audio, linewidth=0.45)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform")
        st.pyplot(fig)
        plt.close(fig)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=320, n_mels=64)
        mel_db = librosa.power_to_db(mel + 1e-10, ref=max)
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=320, x_axis="time", y_axis="mel", ax=ax2)
        ax2.set_title("Mel spectrogram")
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)
        plt.close(fig2)

        if model is not None:
            st.subheader("Candidate classifications")
            results = []
            for idx, segment in enumerate(split_audio(audio, sr, 5.0), start=1):
                features = extract_features(segment, sr)
                candidates = predict(model, features)
                results.append({
                    "Segment": idx,
                    "Top class": candidates[0]["label"],
                    "Model score": candidates[0]["probability"],
                    "Alternatives": ", ".join(
                        f'{c["label"]} ({c["probability"]:.1%})' for c in candidates[:3]
                    )
                })
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="bioacoustic_predictions.csv",
                mime="text/csv",
            )
        else:
            st.subheader("Audio analysis")
            st.write("The waveform and mel spectrogram are shown above. Add labelled training recordings and train the baseline model to enable candidate classification.")
    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
    finally:
        try:
            os_path = Path(temp_path)
            os_path.unlink(missing_ok=True)
        except Exception:
            pass

st.divider()
st.caption("Research prototype only. Validate against independent, expert-labelled recordings before using results for ecological conclusions.")
