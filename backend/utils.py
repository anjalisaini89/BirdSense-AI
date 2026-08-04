from pathlib import Path


def get_audio_files(dataset_path):
    """
    Recursively finds all supported audio files.
    """

    audio_extensions = (".wav", ".mp3", ".flac", ".ogg")

    audio_files = []

    for extension in audio_extensions:
        audio_files.extend(Path(dataset_path).rglob(f"*{extension}"))

    return sorted(audio_files)