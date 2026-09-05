import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score
)


# ============================================================
# BirdSense-AI V2 Evaluation
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR / "features" / "dataset.npz"
LABELS_FILE = BASE_DIR / "features" / "labels.json"
MODEL_FILE = BASE_DIR / "models" / "bird_classifier_v2.keras"

RANDOM_STATE = 42


# ============================================================
# 1. Load dataset
# ============================================================

print("=" * 60)
print("BirdSense-AI V2 Detailed Evaluation")
print("=" * 60)

data = np.load(DATASET_FILE)

X = data["X"]
y = data["y"]

print("\nOriginal dataset:")
print("Features :", X.shape)
print("Labels   :", y.shape)


# ============================================================
# 2. Recreate V2 test split
# ============================================================

X = X[..., np.newaxis]

unique_classes, class_counts = np.unique(
    y,
    return_counts=True
)

evaluation_classes = unique_classes[
    class_counts >= 2
]

singleton_classes = unique_classes[
    class_counts == 1
]

evaluation_mask = np.isin(
    y,
    evaluation_classes
)

singleton_mask = np.isin(
    y,
    singleton_classes
)

X_eval = X[evaluation_mask]
y_eval = y[evaluation_mask]

X_singleton = X[singleton_mask]
y_singleton = y[singleton_mask]


X_train_eval, X_test, y_train_eval, y_test = train_test_split(
    X_eval,
    y_eval,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_eval
)

X_train = np.concatenate(
    [X_train_eval, X_singleton],
    axis=0
)

y_train = np.concatenate(
    [y_train_eval, y_singleton],
    axis=0
)


print("\nRecreated V2 split:")
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))
print("Singleton classes:", len(singleton_classes))


# ============================================================
# 3. Load labels
# ============================================================

with open(LABELS_FILE, "r") as file:
    labels = json.load(file)

labels = {
    int(key): value
    for key, value in labels.items()
}


# ============================================================
# 4. Load V2 model
# ============================================================

print("\nLoading V2 model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded successfully!")


# ============================================================
# 5. Generate predictions
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 6. Overall metrics
# ============================================================

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


print("\n")
print("=" * 60)
print("Overall Metrics")
print("=" * 60)

print(
    f"\nAccuracy   : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1   : {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)


# ============================================================
# 7. Detailed classification report
# ============================================================

test_classes = sorted(
    np.unique(
        np.concatenate(
            [y_test, y_pred]
        )
    )
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

print("\n")
print("=" * 60)
print("Classification Report")
print("=" * 60)

print(report)


# ============================================================
# 8. Save report
# ============================================================

REPORT_FILE = (
    BASE_DIR
    / "models"
    / "evaluation_v2.txt"
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BirdSense-AI V2 Evaluation\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Accuracy    : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Macro F1    : {macro_f1:.4f}\n"
    )

    file.write(
        f"Weighted F1 : {weighted_f1:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(report)


print("\nEvaluation report saved to:")
print(REPORT_FILE)

print("\n" + "=" * 60)
print("Evaluation Complete")
print("=" * 60)