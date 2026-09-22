import sys
import json
import numpy as np
import librosa
import tensorflow as tf

from pathlib import Path

from config import (
    SAMPLE_RATE,
    DURATION,
    N_MFCC,
    N_MELS,
    HOP_LENGTH,
    N_FFT
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "bird_classifier_v6.keras"
LABELS_PATH = BASE_DIR / "features" / "labels.json"


# ============================================================
# Load Model
# ============================================================

print("Loading BirdSense-AI V6 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("V6 model loaded successfully!")


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
# Audio Loading
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
# MFCC Extraction
# ============================================================

def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc


# ============================================================
# Log-Mel Spectrogram
# Used for visualization only
# ============================================================

def extract_log_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel


# ============================================================
# Prepare Visualization Data
# ============================================================

def prepare_visualization(log_mel):

    # --------------------------------------------------------
    # Limit precision to reduce JSON response size
    # --------------------------------------------------------

    spectrogram = np.round(
        log_mel,
        2
    ).tolist()

    # --------------------------------------------------------
    # Time axis
    # --------------------------------------------------------

    time_axis = np.linspace(
        0,
        DURATION,
        log_mel.shape[1]
    )

    # --------------------------------------------------------
    # Frequency axis
    # --------------------------------------------------------

    frequency_axis = librosa.mel_frequencies(
        n_mels=log_mel.shape[0],
        fmin=0,
        fmax=SAMPLE_RATE // 2
    )

    frequency_axis = np.round(
        frequency_axis,
        2
    ).tolist()

    return {
        "spectrogram": spectrogram,
        "time": np.round(time_axis, 3).tolist(),
        "frequency": frequency_axis,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH
    }


# ============================================================
# Bird Prediction
# ============================================================

def predict_bird(file_path):

    print("\nProcessing audio...")

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio = load_audio(file_path)

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = extract_mfcc(audio)

    mfcc_input = mfcc[
        np.newaxis,
        ...,
        np.newaxis
    ]

    # --------------------------------------------------------
    # V6 prediction
    # --------------------------------------------------------

    predictions = model.predict(
        mfcc_input,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_species = labels[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    # --------------------------------------------------------
    # Top 5 predictions
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[-5:][::-1]

    top_predictions = []

    for index in top_indices:

        index = int(index)

        species = labels[index]

        probability = float(
            probabilities[index] * 100
        )

        top_predictions.append(
            {
                "species": species,
                "confidence": round(
                    probability,
                    2
                )
            }
        )

    # --------------------------------------------------------
    # Visualization features
    # --------------------------------------------------------

    log_mel = extract_log_mel(audio)

    visualization = prepare_visualization(
        log_mel
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BirdSense-AI Prediction")
    print("=" * 60)

    print(
        f"\nBird Species : {predicted_species}"
    )

    print(
        f"Confidence   : {confidence:.2f}%"
    )

    print("\nTop 5 Predictions:")
    print("-" * 60)

    for prediction in top_predictions:

        print(
            f"{prediction['species']:<40}"
            f"{prediction['confidence']:.2f}%"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "species": predicted_species,

        "confidence": round(
            confidence,
            2
        ),

        "top_predictions": top_predictions,

        "visualization": visualization
    }


# ============================================================
# Command Line Usage
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            "python predict.py <audio_file>"
        )

        print("\nExample:")

        print(
            "python predict.py "
            "dataset/Voice of Birds/Voice of Birds/"
            "Andean Guan_sound/"
            "Andean Guan10.mp3"
        )

        sys.exit(1)

    audio_file = Path(
        sys.argv[1]
    )

    if not audio_file.exists():

        print(
            "\nError: File not found:"
        )

        print(audio_file)

        sys.exit(1)

    predict_bird(
        audio_file
    )