import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURES_FILE = BASE_DIR / "features" / "dataset.npz"
LABELS_FILE = BASE_DIR / "features" / "labels.json"

MODEL_FILE = BASE_DIR / "models" / "bird_classifier_v6.keras"
OUTPUT_FILE = BASE_DIR / "models" / "evaluation_v6.txt"


# ============================================================
# Load Dataset
# ============================================================

def load_dataset():
    data = np.load(FEATURES_FILE)

    X = data["X"]
    y = data["y"]

    print("=" * 60)
    print("Original dataset:")
    print("=" * 60)

    print(f"Features : {X.shape}")
    print(f"Labels   : {y.shape}")

    return X, y


# ============================================================
# Prepare Test Split
# ============================================================

def prepare_test_data(X, y):

    # Find classes with at least 2 samples
    class_counts = np.bincount(y)

    evaluation_classes = np.where(class_counts >= 2)[0]
    singleton_classes = np.where(class_counts < 2)[0]

    evaluation_mask = np.isin(y, evaluation_classes)
    singleton_mask = np.isin(y, singleton_classes)

    X_eval = X[evaluation_mask]
    y_eval = y[evaluation_mask]

    X_singleton = X[singleton_mask]
    y_singleton = y[singleton_mask]

    # --------------------------------------------------------
    # Same test split methodology used for V3
    # --------------------------------------------------------

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_eval,
        y_eval,
        test_size=0.20,
        random_state=42,
        stratify=y_eval
    )

    # Singleton classes are not included in the test set
    # because they cannot be stratified.
    #
    # They were included in training during V3/V6.
    # We don't need them here for evaluation.

    print("\nTest dataset:")
    print(f"Testing samples : {len(X_test)}")
    print(f"Test classes    : {len(np.unique(y_test))}")

    return X_test, y_test


# ============================================================
# Load Labels
# ============================================================

def load_labels():

    with open(LABELS_FILE, "r") as file:
        labels = json.load(file)

    labels = {
        int(key): value
        for key, value in labels.items()
    }

    return labels


# ============================================================
# Evaluate Model
# ============================================================

def evaluate():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, y = load_dataset()

    # --------------------------------------------------------
    # Prepare test data
    # --------------------------------------------------------

    X_test, y_test = prepare_test_data(X, y)

    # --------------------------------------------------------
    # Add channel dimension
    # --------------------------------------------------------

    X_test = X_test[..., np.newaxis]

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    labels = load_labels()

    # --------------------------------------------------------
    # Load V6 model
    # --------------------------------------------------------

    print("\nLoading V6 model...")

    model = tf.keras.models.load_model(MODEL_FILE)

    print("V6 model loaded successfully!")

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test,
        verbose=1
    )

    predicted_indices = np.argmax(
        predictions,
        axis=1
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = np.mean(
        predicted_indices == y_test
    )

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_test,
        predicted_indices,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Weighted F1
    # --------------------------------------------------------

    weighted_f1 = f1_score(
        y_test,
        predicted_indices,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Convert labels to species names
    # --------------------------------------------------------

    target_names = [
        labels[index]
        for index in sorted(np.unique(y_test))
    ]

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predicted_indices,
        labels=sorted(np.unique(y_test)),
        target_names=target_names,
        zero_division=0
    )

    # ========================================================
    # Display Results
    # ========================================================

    print("\n" + "=" * 60)
    print("V6 Evaluation Results")
    print("=" * 60)

    print(f"\nAccuracy   : {accuracy * 100:.2f}%")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n" + "=" * 60)
    print("Per-Species Classification Report")
    print("=" * 60)

    print(report)

    # ========================================================
    # Save Report
    # ========================================================

    report_text = f"""
============================================================
BirdSense-AI V6 Evaluation
============================================================

Original Dataset
----------------
Features : {X.shape}
Labels   : {y.shape}

Test Dataset
------------
Testing samples : {len(X_test)}
Test classes    : {len(np.unique(y_test))}

Model
-----
{MODEL_FILE}

Evaluation Results
------------------
Accuracy   : {accuracy * 100:.2f}%
Macro F1   : {macro_f1:.4f}
Weighted F1: {weighted_f1:.4f}

============================================================
Per-Species Classification Report
============================================================

{report}

============================================================
Summary
============================================================

Accuracy   : {accuracy * 100:.2f}%
Macro F1   : {macro_f1:.4f}
Weighted F1: {weighted_f1:.4f}

============================================================
"""

    with open(OUTPUT_FILE, "w") as file:
        file.write(report_text)

    print("\nEvaluation report saved to:")
    print(OUTPUT_FILE)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate()