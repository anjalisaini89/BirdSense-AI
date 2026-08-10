import json

import numpy as np
import librosa
from tqdm import tqdm

from config import (
    DATASET_DIR,
    FEATURES_DIR,
    SAMPLE_RATE,
    DURATION,
    N_MFCC
)

from utils import get_audio_files


def load_audio(file_path):
    """Load and standardize an audio file to a fixed duration."""

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
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


def extract_mfcc(audio):
    """Extract MFCC features from audio."""

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc


def build_dataset():

    files = get_audio_files(DATASET_DIR)

    X = []
    y = []

    labels = {}
    failed_files = []

    current_label = 0

    print("=" * 60)
    print("BirdSense-AI Feature Extraction")
    print("=" * 60)

    print(f"\nFound {len(files)} audio files.")
    print("Extracting MFCC features...\n")

    for file in tqdm(files):

        try:

            species = file.parent.name

            if species not in labels:
                labels[species] = current_label
                current_label += 1

            audio = load_audio(file)

            mfcc = extract_mfcc(audio)

            X.append(mfcc)
            y.append(labels[species])

        except Exception as error:

            failed_files.append({
                "file": str(file),
                "error": str(error)
            })

            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    FEATURES_DIR.mkdir(exist_ok=True)

    # Save extracted features
    np.savez(
        FEATURES_DIR / "dataset.npz",
        X=X,
        y=y
    )

    # Save label mapping
    label_mapping = {
        str(label): species
        for species, label in labels.items()
    }

    with open(
        FEATURES_DIR / "labels.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label_mapping,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save failed files
    with open(
        FEATURES_DIR / "failed_files.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            failed_files,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print("=" * 60)

    print(f"Successfully processed : {len(X)}")
    print(f"Failed files           : {len(failed_files)}")
    print(f"Species                : {len(labels)}")

    print(f"\nFeatures shape : {X.shape}")
    print(f"Labels shape   : {y.shape}")

    print("\nSaved files:")

    print(FEATURES_DIR / "dataset.npz")
    print(FEATURES_DIR / "labels.json")
    print(FEATURES_DIR / "failed_files.json")


if __name__ == "__main__":
    build_dataset()