"""
This script is intended as a simple tool for use with Google Colab platform

It's intended use is as follows
`from kerasworkbench.common_imports import *`

This is not intended for use in production software and is only intended to
simplify setup cells in colab.
"""


import numpy as np
import scipy as sp
import scipy.signal as signal
from scipy.signal import (
    periodogram,
)

import librosa
from librosa import (
    amplitude_to_db,
    stft,
    istft,
    load,
)

from librosa.display import (
    specshow
)
