import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

DATASET_FILE = FEATURES_DIR / "dataset.npz"

MODEL_FILE = MODELS_DIR / "bird_classifier_v3.keras"
HISTORY_FILE = MODELS_DIR / "training_history_v3.json"


# ============================================================
# Training configuration
# ============================================================

EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

def load_dataset():

    print("=" * 60)
    print("BirdSense-AI V3 Dataset")
    print("=" * 60)

    data = np.load(DATASET_FILE)

    X = data["X"]
    y = data["y"]

    print("\nFeatures :", X.shape)
    print("Labels   :", y.shape)

    return X, y


# ============================================================
# Prepare train / validation / test data
# ============================================================

def prepare_data(X, y):

    print("\nPreparing dataset...")

    X = X[..., np.newaxis]

    num_classes = len(np.unique(y))

    print("Number of classes:", num_classes)

    unique_classes, class_counts = np.unique(
        y,
        return_counts=True
    )

    singleton_classes = unique_classes[
        class_counts == 1
    ]

    evaluation_classes = unique_classes[
        class_counts >= 2
    ]

    print(
        "Classes with >=2 samples :",
        len(evaluation_classes)
    )

    print(
        "Singleton classes        :",
        len(singleton_classes)
    )

    # --------------------------------------------------------
    # Separate singleton classes
    # They MUST remain in training.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # First split:
    # evaluation classes → train/test
    # --------------------------------------------------------

    (
        X_train_eval,
        X_test,
        y_train_eval,
        y_test
    ) = train_test_split(
        X_eval,
        y_eval,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_eval
    )

    # --------------------------------------------------------
    # Second split:
    # training portion → train/validation
    #
    # Important:
    # validation is created BEFORE singleton samples
    # are added to training.
    # --------------------------------------------------------

    (
        X_train_eval,
        X_val,
        y_train_eval,
        y_val
    ) = train_test_split(
        X_train_eval,
        y_train_eval,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_eval
    )

    # --------------------------------------------------------
    # Add singleton classes ONLY to training
    # --------------------------------------------------------

    X_train = np.concatenate(
        [
            X_train_eval,
            X_singleton
        ],
        axis=0
    )

    y_train = np.concatenate(
        [
            y_train_eval,
            y_singleton
        ],
        axis=0
    )

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = rng.permutation(
        len(X_train)
    )

    X_train = X_train[indices]
    y_train = y_train[indices]

    print("\nDataset split:")
    print(
        "Training samples   :",
        len(X_train)
    )

    print(
        "Validation samples :",
        len(X_val)
    )

    print(
        "Testing samples    :",
        len(X_test)
    )

    print(
        "Singleton training :",
        len(X_singleton)
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        num_classes
    )


# ============================================================
# Calculate class weights
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
        "Class weights calculated for",
        len(class_weights),
        "classes."
    )

    return class_weights


# ============================================================
# Build CNN
# ============================================================

def build_model(input_shape, num_classes):

    print("\nBuilding CNN model...")

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

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

        tf.keras.layers.Dropout(
            0.25
        ),

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

        tf.keras.layers.Dropout(
            0.25
        ),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.4
        ),

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
# Train
# ============================================================

def train():

    X, y = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        num_classes
    ) = prepare_data(
        X,
        y
    )

    # --------------------------------------------------------
    # One-hot encode labels
    # --------------------------------------------------------

    y_train_encoded = tf.keras.utils.to_categorical(
        y_train,
        num_classes=num_classes
    )

    y_val_encoded = tf.keras.utils.to_categorical(
        y_val,
        num_classes=num_classes
    )

    y_test_encoded = tf.keras.utils.to_categorical(
        y_test,
        num_classes=num_classes
    )

    # --------------------------------------------------------
    # Class weights
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
    # Callbacks
    # --------------------------------------------------------

    checkpoint = tf.keras.callbacks.ModelCheckpoint(

        MODEL_FILE,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    )

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
    print("Starting V3 Training")
    print("=" * 60)

    history = model.fit(

        X_train,

        y_train_encoded,

        validation_data=(
            X_val,
            y_val_encoded
        ),

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
    # Final test evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("Evaluating V3 Model")
    print("=" * 60)

    loss, accuracy = model.evaluate(

        X_test,

        y_test_encoded,

        verbose=1
    )

    print("\n")
    print("=" * 60)
    print("BirdSense-AI V3 Results")
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

    print("\nV3 model saved to:")
    print(MODEL_FILE)

    print("\nTraining history saved to:")
    print(HISTORY_FILE)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train()