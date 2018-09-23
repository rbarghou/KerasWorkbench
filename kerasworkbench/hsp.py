import numpy as np


def hsp(spectrogram, num_harmonics):
    return np.array([
        np.product(
            np.abs(spectrogram)[::f0, :][:num_harmonics, :],
            axis=0
        )
        for f0 in range(1, spectrogram.shape[0] + 1)
    ])
