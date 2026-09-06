import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR / "features" / "dataset.npz"
LABELS_FILE = BASE_DIR / "features" / "labels.json"
MODEL_FILE = BASE_DIR / "models" / "bird_classifier_v3.keras"
OUTPUT_FILE = BASE_DIR / "models" / "evaluation_v3.txt"

RANDOM_STATE = 42


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("=" * 60)
print("BirdSense-AI V3 Evaluation")
print("=" * 60)

data = np.load(DATASET_FILE)

X = data["X"]
y = data["y"]

print("\nOriginal dataset:")
print("Features :", X.shape)
print("Labels   :", y.shape)


# CNN expects channel dimension
X = X[..., np.newaxis]


# --------------------------------------------------
# Recreate V3 test split
# --------------------------------------------------

unique_classes, class_counts = np.unique(
    y,
    return_counts=True
)

evaluation_classes = unique_classes[class_counts >= 2]

evaluation_mask = np.isin(
    y,
    evaluation_classes
)

X_eval = X[evaluation_mask]
y_eval = y[evaluation_mask]

_, X_test, _, y_test = train_test_split(
    X_eval,
    y_eval,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_eval
)

print("\nTest dataset:")
print("Testing samples :", len(X_test))
print("Test classes    :", len(np.unique(y_test)))


# --------------------------------------------------
# Load labels
# --------------------------------------------------

with open(LABELS_FILE, "r") as file:
    labels = json.load(file)

labels = {
    int(key): value
    for key, value in labels.items()
}


# --------------------------------------------------
# Load V3 model
# --------------------------------------------------

print("\nLoading V3 model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("V3 model loaded successfully!")


# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("V3 Evaluation Results")
print("=" * 60)

print(f"\nAccuracy   : {accuracy * 100:.2f}%")
print(f"Macro F1   : {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

test_classes = np.unique(
    np.concatenate([y_test, y_pred])
)

target_names = [
    labels[index]
    for index in test_classes
]

report = classification_report(
    y_test,
    y_pred,
    labels=test_classes,
    target_names=target_names,
    zero_division=0
)

print("\n" + "=" * 60)
print("Per-Species Classification Report")
print("=" * 60)

print(report)


# --------------------------------------------------
# Save evaluation report
# --------------------------------------------------

with open(OUTPUT_FILE, "w") as file:

    file.write("=" * 60 + "\n")
    file.write("BirdSense-AI V3 Evaluation\n")
    file.write("=" * 60 + "\n\n")

    file.write(
        f"Test Samples : {len(X_test)}\n"
    )

    file.write(
        f"Test Classes  : {len(np.unique(y_test))}\n\n"
    )

    file.write(
        f"Accuracy      : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Macro F1      : {macro_f1:.4f}\n"
    )

    file.write(
        f"Weighted F1   : {weighted_f1:.4f}\n\n"
    )

    file.write("=" * 60 + "\n")
    file.write("Classification Report\n")
    file.write("=" * 60 + "\n\n")

    file.write(report)

print("\nEvaluation report saved to:")
print(OUTPUT_FILE)