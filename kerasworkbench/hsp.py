import numpy as np


def hsp(spectrogram, num_harmonics):
    """Harmonic Spectrum Product
    Returns an array where every frequency bin is the product of it's harmonics up to num_harmonics

    This function is useful for trying to identify the fundamental frequencies in an audio sample where
    harmonics are present.  It will not identify the specific harmonic, but where harmonics are present
    this function should amplify the fundamentals.  Where they are absent, the signal should be dampened.

    :param spectrogram: 2d array of real or complex numbers of the shape [n_frequency_bins x n_frames]
    :param num_harmonics: The number of harmonics to consider in the HSP analysis
    :return result: A real array with the same shape as the spectrogram
    """
    return np.array([
        np.product(
            np.abs(spectrogram)[::f0, :][:num_harmonics, :],
            axis=0
        )
        for f0 in range(1, spectrogram.shape[0] + 1)
    ])
