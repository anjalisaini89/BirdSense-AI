import numpy as np
import librosa

from pathlib import Path
from tqdm import tqdm

from config import DATASET_DIR, FEATURES_DIR, SAMPLE_RATE, DURATION, N_MFCC
from utils import get_audio_files

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

def extract_mfcc(audio):

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

    current_label = 0

    for file in tqdm(files):

        species = file.parent.name

        if species not in labels:
            labels[species] = current_label
            current_label += 1

        audio = load_audio(file)

        mfcc = extract_mfcc(audio)

        X.append(mfcc)

        y.append(labels[species])

    X = np.array(X)
    y = np.array(y)

    FEATURES_DIR.mkdir(exist_ok=True)

    np.savez(
        FEATURES_DIR / "dataset.npz",
        X=X,
        y=y
    )

    print("\nSaved dataset!")
    print("Samples :", X.shape)
    print("Labels :", y.shape)

   if __name__ == "__main__":
    build_dataset()