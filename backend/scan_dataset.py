from pathlib import Path
from collections import Counter

from utils import get_audio_files
from config import DATASET_DIR


def scan_dataset():

    files = get_audio_files(DATASET_DIR)

    print("=" * 60)
    print("BirdSense-AI Dataset Scanner")
    print("=" * 60)

    print(f"\nTotal Audio Files : {len(files)}\n")

    species_counter = Counter()

    for file in files:

        # Parent folder name is the bird species
        species = file.parent.name

        species_counter[species] += 1

    print("Bird Species\n")

    for species, count in sorted(species_counter.items()):
        print(f"{species:<35} {count}")

    print("\n" + "=" * 60)
    print(f"Total Species : {len(species_counter)}")
    print("=" * 60)


if __name__ == "__main__":
    scan_dataset()