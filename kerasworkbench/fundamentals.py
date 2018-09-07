"""Functions for extracting fundamentals from audio"""
import numpy as np
from scipy.signal import (
    argrelmax,
    periodogram,
)
import pandas as pd


def get_peak_frequencies(f, Pxx, sort=True):
    """
    Accepts the input from scipy.signal.periodogram as such:
    peak_frequencies = get_peak_frequencies(*periodogram(clip, sr))
    :param f: the frequencies
    :param Pxx: the power level
    :param sort: should the output be sorted
    :return:
    """
    peak_idxs = argrelmax(
        np.maximum(Pxx, np.mean(Pxx))
    )[0]
    peak_fs = f[peak_idxs]
    peak_Pxx = Pxx[peak_idxs]
    peaks = zip(peak_Pxx, peak_fs, peak_idxs)
    if sort:
        peaks = sorted(peaks)
        peaks.reverse()
    return pd.DataFrame(peaks, columns=("Pxx", "fs", "idx" ))


if __name__ == "__main__":
    from unittest import TestCase, main

    class Test(TestCase):

        def test(self):
            sr = 22050
            T = np.linspace(0, 1, sr)
            freqs = [
                220, 330, 123, 654, 444
            ]
            clip = np.zeros_like(T)
            for freq in freqs:
                offset = freq % 29
                power = offset
                clip += power * np.sin((2 * np.pi * T * freq) + offset)

            df = get_peak_frequencies(*periodogram(clip, sr))

            expected_freqs = sorted(set(freqs))
            received_freqs = sorted(set(
                int(freq) for freq in df.fs
            ))

            self.assertEqual(expected_freqs, received_freqs)

    main()
