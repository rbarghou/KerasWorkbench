import numpy as np
import scipy.signal

from .smooting import iterative_convolve


def get_peaks(frame):
    x_abs = np.abs(frame)
    x_conv = iterative_convolve(x_abs, 3)
    x_var = np.sqrt(iterative_convolve((x_abs - x_conv) ** 2, 10))

    _peaks = scipy.signal.argrelmax(x_abs)[0]
    peaks = _peaks[
        (
            x_abs[_peaks]
            >
            (
                x_conv[_peaks]
                +
                x_var[_peaks]
            )
        )
        *
        x_abs[_peaks] > np.mean(x_abs)
    ]
    return peaks
