"""Functions for extracting fundamentals from audio"""
import numpy as np
from scipy.signal import (
    argrelmax,
    periodogram,
)
import pandas as pd

from .smooting import iterative_convolve


def get_peak_frequencies_from_periodogram(f, Pxx, sort=True):
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


def get_good_peaks(x, alpha=.3):
    _peaks = argrelmax(iterative_convolve(np.abs(x), 2))[0]
    good_peaks = _peaks[
        np.abs(np.abs(x[_peaks])) > (alpha * np.mean(np.abs(x[_peaks])))
    ]
    return good_peaks


def harmonic_filter(f0, peaks, alpha=2):
    return peaks[
        np.minimum(
            peaks % f0,
            np.abs((-peaks) % f0),
        ) < alpha
    ]


def search_range_for_f0(low_f0, high_f0, steps, peaks, alpha=4):
    f = np.linspace(low_f0, high_f0, steps)
    n_peaks = [len(harmonic_filter(f0, peaks, alpha)) for f0 in f]
    return f[n_peaks == np.max(n_peaks)][0]


def estimate_f0(x, alpha1=.1, alpha2=4):
    x_peaks = get_good_peaks(x, alpha=alpha1)
    f0_candidate_midpoint = x_peaks[0]
    low = f0_candidate_midpoint - 1
    high = f0_candidate_midpoint + 1
    f0 = search_range_for_f0(low, high, 21, x_peaks, alpha=alpha2)
    return f0


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

            df = get_peak_frequencies_from_periodogram(*periodogram(clip, sr))

            expected_freqs = sorted(set(freqs))
            received_freqs = sorted(set(
                int(freq) for freq in df.fs
            ))

            self.assertEqual(expected_freqs, received_freqs)

    main()
