import numpy as np
import librosa
import json

from pathlib import Path
from tensorflow.keras.models import load_model

from config import SAMPLE_RATE, DURATION, N_MFCC


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path("models/bird_classifier.keras")
LABEL_PATH = Path("features/label_mapping.json")


# ============================================================
# Load Audio
# ============================================================

def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    required_length = SAMPLE_RATE * DURATION

    if len(audio) > required_length:

        audio = audio[:required_length]

    else:

        audio = np.pad(
            audio,
            (0, required_length - len(audio))
        )

    return audio


# ============================================================
# Extract MFCC
# ============================================================

def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc


# ============================================================
# Predict Bird
# ============================================================

def predict_bird(audio_file):

    print("\nLoading model...")

    model = load_model(MODEL_PATH)

    print("Model loaded successfully!")

    # Load labels
    with open(LABEL_PATH, "r") as file:
        label_mapping = json.load(file)

    # Load audio
    print("\nProcessing audio...")

    audio = load_audio(audio_file)

    # Extract MFCC
    mfcc = extract_mfcc(audio)

    # Add batch and channel dimensions
    X = mfcc[np.newaxis, ..., np.newaxis]

    # Prediction
    predictions = model.predict(X, verbose=0)

    predicted_index = np.argmax(predictions[0])

    confidence = predictions[0][predicted_index] * 100

    species = label_mapping[str(predicted_index)]

    print("\n" + "=" * 60)
    print("BirdSense-AI Prediction")
    print("=" * 60)

    print(f"\nPredicted Bird : {species}")
    print(f"Confidence     : {confidence:.2f}%")

    print("=" * 60)

    return species, confidence


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    audio_file = input(
        "\nEnter path to bird audio file: "
    ).strip()

    audio_path = Path(audio_file)

    if not audio_path.exists():

        print("\nError: Audio file not found.")

    else:

        predict_bird(audio_path)