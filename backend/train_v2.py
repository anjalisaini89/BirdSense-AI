import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# BirdSense-AI V2 Training Pipeline
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

DATASET_FILE = FEATURES_DIR / "dataset.npz"
LABELS_FILE = FEATURES_DIR / "labels.json"

MODEL_FILE = MODELS_DIR / "bird_classifier_v2.keras"
HISTORY_FILE = MODELS_DIR / "training_history_v2.json"

EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42


# ============================================================
# 1. Load Dataset
# ============================================================

def load_dataset():

    print("=" * 60)
    print("BirdSense-AI V2 Dataset")
    print("=" * 60)

    data = np.load(DATASET_FILE)

    X = data["X"]
    y = data["y"]

    print(f"\nFeatures : {X.shape}")
    print(f"Labels   : {y.shape}")

    return X, y


# ============================================================
# 2. Prepare Dataset
# ============================================================

def prepare_data(X, y):

    print("\nPreparing data...")

    # CNN expects:
    # (samples, height, width, channels)

    X = X[..., np.newaxis]

    num_classes = len(np.unique(y))

    print(f"Number of classes: {num_classes}")

    # --------------------------------------------------------
    # Split classes with enough samples from singleton classes
    # --------------------------------------------------------

    unique_classes, class_counts = np.unique(y, return_counts=True)

    evaluation_classes = unique_classes[class_counts >= 2]
    singleton_classes = unique_classes[class_counts == 1]

    print(
        f"Classes with >=2 samples : {len(evaluation_classes)}"
    )

    print(
        f"Singleton classes         : {len(singleton_classes)}"
    )

    # --------------------------------------------------------
    # Evaluation-eligible samples
    # --------------------------------------------------------

    evaluation_mask = np.isin(y, evaluation_classes)

    X_eval = X[evaluation_mask]
    y_eval = y[evaluation_mask]

    # Singleton samples remain in training
    singleton_mask = np.isin(y, singleton_classes)

    X_singleton = X[singleton_mask]
    y_singleton = y[singleton_mask]

    # --------------------------------------------------------
    # Split evaluation-eligible samples
    # --------------------------------------------------------

    X_train_eval, X_test, y_train_eval, y_test = train_test_split(
        X_eval,
        y_eval,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_eval
    )

    # --------------------------------------------------------
    # Add singleton classes back to training
    # --------------------------------------------------------

    X_train = np.concatenate(
        [X_train_eval, X_singleton],
        axis=0
    )

    y_train = np.concatenate(
        [y_train_eval, y_singleton],
        axis=0
    )

    print("\nDataset split:")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test, num_classes


# ============================================================
# 3. Calculate Class Weights
# ============================================================

def calculate_class_weights(y_train):

    print("\nCalculating class weights...")

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = dict(
        zip(classes, weights)
    )

    print(
        f"Class weights calculated for "
        f"{len(class_weights)} classes."
    )

    return class_weights


# ============================================================
# 4. Build CNN
# ============================================================

def build_model(input_shape, num_classes):

    print("\nBuilding CNN model...")

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

        # ----------------------------------------------------
        # Convolution Block 1
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Dropout(0.25),

        # ----------------------------------------------------
        # Convolution Block 2
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Dropout(0.25),

        # ----------------------------------------------------
        # Convolution Block 3
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling2D(),

        # ----------------------------------------------------
        # Dense Layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.4),

        # ----------------------------------------------------
        # Output Layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ============================================================
# 5. Train Model
# ============================================================

def train():

    X, y = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        num_classes
    ) = prepare_data(X, y)

    # --------------------------------------------------------
    # One-hot encode labels
    # --------------------------------------------------------

    y_train_encoded = tf.keras.utils.to_categorical(
        y_train,
        num_classes=num_classes
    )

    y_test_encoded = tf.keras.utils.to_categorical(
        y_test,
        num_classes=num_classes
    )

    # --------------------------------------------------------
    # Calculate class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        y_train
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        X_train.shape[1:],
        num_classes
    )

    print("\nModel:")
    model.summary()

    MODELS_DIR.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = tf.keras.callbacks.ModelCheckpoint(

        MODEL_FILE,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=7,

        restore_best_weights=True,

        verbose=1
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("Starting V2 Training")
    print("=" * 60)

    history = model.fit(

        X_train,

        y_train_encoded,

        validation_split=0.20,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=[
            checkpoint,
            early_stopping
        ],

        verbose=1
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("Evaluating V2 Model")
    print("=" * 60)

    loss, accuracy = model.evaluate(
        X_test,
        y_test_encoded,
        verbose=1
    )

    print("\n")
    print("=" * 60)
    print("BirdSense-AI V2 Results")
    print("=" * 60)

    print(
        f"\nTest Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Test Loss     : {loss:.4f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with open(
        HISTORY_FILE,
        "w"
    ) as file:

        json.dump(
            history.history,
            file,
            indent=4
        )

    print("\nV2 model saved to:")
    print(MODEL_FILE)

    print("\nTraining history saved to:")
    print(HISTORY_FILE)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train()