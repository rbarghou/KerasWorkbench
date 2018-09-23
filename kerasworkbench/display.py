import librosa
import librosa.display
import numpy as np


def spec(S):
    librosa.display.specshow(
        librosa.amplitude_to_db(
            np.abs(S),
            ref=np.max
        ),
        y_axis="log",
        x_axis="time",
    )