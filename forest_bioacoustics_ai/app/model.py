from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

MODEL_PATH = Path("models/bioacoustic_model.joblib")

def train_model(X: np.ndarray, y: np.ndarray):
    """Train a simple baseline classifier from labelled feature vectors."""
    labels = np.unique(y)
    if len(labels) < 2:
        raise ValueError("Training requires at least two distinct classes.")
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    )
    model.fit(X, y)
    return model

def save_model(model, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)

def predict(model, features: np.ndarray):
    """Return sorted class probabilities for a single feature vector."""
    if model is None:
        raise RuntimeError("No trained model found. Train the model first.")
    probs = model.predict_proba(features.reshape(1, -1))[0]
    classes = model.classes_
    order = np.argsort(probs)[::-1]
    return [{"label": str(classes[i]), "probability": float(probs[i])} for i in order]
