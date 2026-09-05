import json
import numpy as np
import librosa

from pathlib import Path
from tqdm import tqdm

from config import DATASET_DIR, FEATURES_DIR, SAMPLE_RATE, DURATION, N_MFCC
from utils import get_audio_files


def load_audio(file_path):
    """
    Load an audio file and make it exactly DURATION seconds long.
    """

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


def extract_mfcc(audio):
    """
    Extract MFCC features from an audio signal.
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc


def build_dataset():
    """
    Extract MFCC features from all bird audio files
    and save the resulting dataset and label mapping.
    """

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

            # Bird species is the parent folder name
            species = file.parent.name

            # Assign a unique numerical label
            if species not in labels:
                labels[species] = current_label
                current_label += 1

            # Load audio
            audio = load_audio(file)

            # Extract MFCC
            mfcc = extract_mfcc(audio)

            # Store features and label
            X.append(mfcc)
            y.append(labels[species])

        except Exception as error:

            print(f"\nFailed to process: {file}")
            print(f"Error: {repr(error)}")

            failed_files.append({
                "file": str(file),
                "error": repr(error)
            })

            continue

    # Convert lists to NumPy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Create features directory
    FEATURES_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Save extracted features
    # --------------------------------------------------

    np.savez(
        FEATURES_DIR / "dataset.npz",
        X=X,
        y=y
    )

    # --------------------------------------------------
    # Save label mapping
    # --------------------------------------------------

    label_mapping = {
        str(label): species
        for species, label in labels.items()
    }

    with open(FEATURES_DIR / "labels.json", "w") as f:
        json.dump(label_mapping, f, indent=4)

    # --------------------------------------------------
    # Save failed files
    # --------------------------------------------------

    if failed_files:

        with open(FEATURES_DIR / "failed_files.json", "w") as f:
            json.dump(failed_files, f, indent=4)

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Feature Extraction Complete")
    print("=" * 60)

    print(f"Total audio files : {len(files)}")
    print(f"Successful files  : {len(X)}")
    print(f"Failed files      : {len(failed_files)}")
    print(f"Number of species : {len(labels)}")

    print(f"\nFeature shape     : {X.shape}")
    print(f"Label shape       : {y.shape}")

    print("\nSaved files:")
    print(f"  {FEATURES_DIR / 'dataset.npz'}")
    print(f"  {FEATURES_DIR / 'labels.json'}")

    if failed_files:
        print(f"  {FEATURES_DIR / 'failed_files.json'}")

    print("=" * 60)


if __name__ == "__main__":
    build_dataset()