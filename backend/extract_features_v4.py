import json
import numpy as np
import librosa

from pathlib import Path
from tqdm import tqdm

from config import (
    DATASET_DIR,
    FEATURES_DIR,
    SAMPLE_RATE,
    DURATION,
    N_MELS,
    HOP_LENGTH,
    N_FFT
)

from utils import get_audio_files


# --------------------------------------------------
# Audio Loading
# --------------------------------------------------

def load_audio(file_path):
    """
    Load an audio file and make it exactly
    DURATION seconds long.
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


# --------------------------------------------------
# Log-Mel Spectrogram
# --------------------------------------------------

def extract_log_mel(audio):
    """
    Extract a log-Mel spectrogram from
    an audio signal.
    """

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


# --------------------------------------------------
# Build Dataset
# --------------------------------------------------

def build_dataset():

    files = get_audio_files(DATASET_DIR)

    X = []
    y = []

    labels = {}
    failed_files = []

    current_label = 0

    print("=" * 60)
    print("BirdSense-AI V4 Feature Extraction")
    print("=" * 60)

    print(f"\nFound {len(files)} audio files.")
    print("Extracting Log-Mel Spectrogram features...\n")


    for file in tqdm(files):

        try:

            # Bird species = parent folder name
            species = file.parent.name

            # Assign numerical label
            if species not in labels:

                labels[species] = current_label
                current_label += 1

            # Load audio
            audio = load_audio(file)

            # Extract Log-Mel Spectrogram
            log_mel = extract_log_mel(audio)

            # Store feature and label
            X.append(log_mel)
            y.append(labels[species])

        except Exception as error:

            print(f"\nFailed to process: {file}")
            print(f"Error: {repr(error)}")

            failed_files.append({
                "file": str(file),
                "error": repr(error)
            })

            continue


    # --------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )


    # --------------------------------------------------
    # Save V4 dataset
    # --------------------------------------------------

    FEATURES_DIR.mkdir(
        exist_ok=True
    )

    np.savez(
        FEATURES_DIR / "dataset_v4.npz",
        X=X,
        y=y
    )


    # --------------------------------------------------
    # Save labels
    # --------------------------------------------------

    label_mapping = {
        str(label): species
        for species, label in labels.items()
    }

    with open(
        FEATURES_DIR / "labels_v4.json",
        "w"
    ) as f:

        json.dump(
            label_mapping,
            f,
            indent=4
        )


    # --------------------------------------------------
    # Save failed files
    # --------------------------------------------------

    if failed_files:

        with open(
            FEATURES_DIR / "failed_files_v4.json",
            "w"
        ) as f:

            json.dump(
                failed_files,
                f,
                indent=4
            )


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("V4 Feature Extraction Complete")
    print("=" * 60)

    print(f"Total audio files : {len(files)}")
    print(f"Successful files  : {len(X)}")
    print(f"Failed files      : {len(failed_files)}")
    print(f"Number of species : {len(labels)}")

    print(f"\nFeature shape     : {X.shape}")
    print(f"Label shape       : {y.shape}")

    print("\nSaved files:")

    print(
        f"  {FEATURES_DIR / 'dataset_v4.npz'}"
    )

    print(
        f"  {FEATURES_DIR / 'labels_v4.json'}"
    )

    if failed_files:

        print(
            f"  {FEATURES_DIR / 'failed_files_v4.json'}"
        )

    print("=" * 60)


if __name__ == "__main__":
    build_dataset()