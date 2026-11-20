
from pathlib import Path
from ultralytics import YOLO
import torch
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    """Load the YOLO model once and reuse it."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return YOLO(str(MODEL_PATH))


def detect_potholes(image, confidence=0.10):
    """Detect potholes in an image or video frame."""

    model = load_model()

    results = model.predict(
        source=image,
        conf=confidence,
        device=DEVICE,
        verbose=False
    )

    return results[0]
