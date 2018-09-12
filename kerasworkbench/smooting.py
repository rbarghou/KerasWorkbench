import numpy as np


def iterative_convolve(x, n):
    _x = x.copy()
    for _ in range(n):
        _x = np.convolve(_x, np.ones(3.0) / 3., "same")
    return _x
