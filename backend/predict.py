import sys
import json
import numpy as np
import librosa
import tensorflow as tf

from pathlib import Path

from config import SAMPLE_RATE, DURATION, N_MFCC


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "bird_classifier.keras"
LABELS_PATH = BASE_DIR / "features" / "labels.json"


# ============================================================
# Load Model
# ============================================================

print("Loading BirdSense-AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# Load Labels
# ============================================================

with open(LABELS_PATH, "r") as file:
    labels = json.load(file)

labels = {
    int(key): value
    for key, value in labels.items()
}


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

def predict_bird(file_path):

    print("\nProcessing audio...")

    # Load audio
    audio = load_audio(file_path)

    # Extract MFCC
    mfcc = extract_mfcc(audio)

    # Add batch and channel dimensions
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # Model prediction
    predictions = model.predict(
        mfcc,
        verbose=0
    )

    probabilities = predictions[0]

    # Get predicted class
    predicted_index = int(np.argmax(probabilities))

    predicted_species = labels[predicted_index]

    confidence = float(probabilities[predicted_index] * 100)

    # ========================================================
    # Top 5 Predictions
    # ========================================================

    top_indices = np.argsort(probabilities)[-5:][::-1]

    top_predictions = []

    for index in top_indices:

        index = int(index)

        species = labels[index]

        probability = float(probabilities[index] * 100)

        top_predictions.append({
            "species": species,
            "confidence": round(probability, 2)
        })

    # ========================================================
    # Console Output
    # ========================================================

    print("\n" + "=" * 60)
    print("BirdSense-AI Prediction")
    print("=" * 60)

    print(f"\nBird Species : {predicted_species}")
    print(f"Confidence   : {confidence:.2f}%")

    print("\nTop 5 Predictions:")
    print("-" * 60)

    for prediction in top_predictions:

        print(
            f"{prediction['species']:<40} "
            f"{prediction['confidence']:.2f}%"
        )

    print("=" * 60)

    # ========================================================
    # RETURN RESULT TO FASTAPI
    # ========================================================

    return {
        "species": predicted_species,
        "confidence": round(confidence, 2),
        "top_predictions": top_predictions
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print("python predict.py <audio_file>")

        print("\nExample:")
        print(
            "python predict.py "
            "dataset/Voice of Birds/Voice of Birds/"
            "Andean Guan_sound/Andean Guan10.mp3"
        )

        sys.exit(1)

    audio_file = Path(sys.argv[1])

    if not audio_file.exists():

        print("\nError: File not found:")
        print(audio_file)

        sys.exit(1)

    predict_bird(audio_file)