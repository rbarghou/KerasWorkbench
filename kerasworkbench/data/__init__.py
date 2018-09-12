import os

import librosa


SUPPORTED_FORMATS = ("mp3")

DATA_DIR = os.path.dirname(__file__)
AUDIO_DATA_DIR = os.path.join(DATA_DIR, "audio")

AUDIO_FILES = sorted([
    fn for fn in os.listdir(AUDIO_DATA_DIR)
    if any(fn.endswith(format) for format in SUPPORTED_FORMATS)
])


def load_audio_file(i, *args, **kwargs):
    return librosa.load(os.path.join(AUDIO_DATA_DIR, AUDIO_FILES[i]), *args, **kwargs)
