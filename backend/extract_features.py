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