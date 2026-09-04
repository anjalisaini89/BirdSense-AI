import librosa
from pathlib import Path

from config import SAMPLE_RATE, DURATION


files = [
    Path(
        r"dataset/Voice of Birds/Voice of Birds/"
        r"Darwins Nothura_sound/Darwins Nothura21.mp3"
    ),

    Path(
        r"dataset/Voice of Birds/Voice of Birds/"
        r"Southern Cassowary_sound/Southern Cassowary2.mp3"
    ),

    Path(
        r"dataset/Voice of Birds/Voice of Birds/"
        r"Spotted Nothura_sound/Spotted Nothura2.mp3"
    )
]


for file_path in files:

    print("\n" + "=" * 60)
    print(f"Testing: {file_path.name}")
    print("=" * 60)

    try:

        print("File exists:", file_path.exists())
        print("File size:", file_path.stat().st_size, "bytes")

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE
        )

        duration = len(audio) / sr

        print("Loaded successfully!")
        print("Sample rate:", sr)
        print("Audio samples:", len(audio))
        print(f"Duration: {duration:.2f} seconds")

    except Exception as error:

        print("FAILED!")
        print("Error type:", type(error).__name__)
        print("Error:", repr(error))