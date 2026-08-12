import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split


FEATURES_DIR = Path("features")
MODELS_DIR = Path("models")

DATASET_FILE = FEATURES_DIR / "dataset.npz"

EPOCHS = 30
BATCH_SIZE = 32


def load_dataset():

    data = np.load(DATASET_FILE)

    X = data["X"]
    y = data["y"]

    print("=" * 60)
    print("BirdSense-AI Dataset")
    print("=" * 60)

    print("Features :", X.shape)
    print("Labels   :", y.shape)

    return X, y


def prepare_data(X, y):

    # Add channel dimension for CNN
    X = X[..., np.newaxis]

    num_classes = len(np.unique(y))

    y_encoded = tf.keras.utils.to_categorical(
        y,
        num_classes=num_classes
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, num_classes


def build_model(input_shape, num_classes):

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

        tf.keras.layers.Dropout(0.25),

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

        tf.keras.layers.Dropout(0.4),

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


def train():

    X, y = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        num_classes
    ) = prepare_data(X, y)

    print("\nTraining samples :", len(X_train))
    print("Testing samples  :", len(X_test))
    print("Number of classes:", num_classes)

    model = build_model(
        X_train.shape[1:],
        num_classes
    )

    print("\nModel:")
    model.summary()

    MODELS_DIR.mkdir(exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODELS_DIR / "bird_classifier.keras",
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

    print("\nStarting training...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            checkpoint,
            early_stopping
        ],
        verbose=1
    )

    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print("\n" + "=" * 60)
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")
    print("=" * 60)

    with open(
        MODELS_DIR / "training_history.json",
        "w"
    ) as f:

        json.dump(
            history.history,
            f,
            indent=4
        )

    print("\nModel saved:")
    print(MODELS_DIR / "bird_classifier.keras")


if __name__ == "__main__":
    train()