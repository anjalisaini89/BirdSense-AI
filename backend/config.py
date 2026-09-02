from pathlib import Path

# ===============================
# Project Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"

# ===============================
# Audio Parameters
# ===============================

SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# ===============================
# Training Parameters
# ===============================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_STATE = 42