import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR / "features" / "dataset.npz"
MODEL_FILE = BASE_DIR / "models" / "bird_classifier.keras"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("=" * 60)
print("BirdSense-AI Model Evaluation")
print("=" * 60)

print("\nLoading dataset...")

data = np.load(DATASET_FILE)

X = data["X"]
y = data["y"]

print(f"Features : {X.shape}")
print(f"Labels   : {y.shape}")


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

X = X[..., np.newaxis]

num_classes = len(np.unique(y))

y_encoded = tf.keras.utils.to_categorical(
    y,
    num_classes=num_classes
)


# --------------------------------------------------
# Recreate the original test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\nDataset split:")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Number of classes: {num_classes}")


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_FILE)

print("Model loaded successfully!")


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

print("\nEvaluating model on held-out test set...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("BirdSense-AI Evaluation Results")
print("=" * 60)

print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

print("=" * 60)