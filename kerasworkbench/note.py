from numpy import log2, power


A4 = 440
C0 = A4 * power(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note(freq):
    h = int(round(12 * log2(freq / C0)))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)
