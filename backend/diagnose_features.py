import numpy as np

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = BASE_DIR / "features"


def inspect_dataset(path, name):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    data = np.load(path)

    X = data["X"]
    y = data["y"]

    print("\nShape:")
    print(X.shape)

    print("\nData type:")
    print(X.dtype)

    print("\nMinimum:")
    print(X.min())

    print("\nMaximum:")
    print(X.max())

    print("\nMean:")
    print(X.mean())

    print("\nStandard deviation:")
    print(X.std())

    print("\nNumber of NaN values:")
    print(np.isnan(X).sum())

    print("\nNumber of infinite values:")
    print(np.isinf(X).sum())

    print("\nFirst sample statistics:")

    sample = X[0]

    print("Shape:", sample.shape)
    print("Min  :", sample.min())
    print("Max  :", sample.max())
    print("Mean :", sample.mean())
    print("Std  :", sample.std())

    print("\nLabels:")
    print("Number of samples:", len(y))
    print("Number of classes:", len(np.unique(y)))


inspect_dataset(
    FEATURES_DIR / "dataset.npz",
    "V1/V3 MFCC DATASET"
)

inspect_dataset(
    FEATURES_DIR / "dataset_v4.npz",
    "V4 LOG-MEL DATASET"
)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)