import numpy as np
from scipy.signal import find_peaks

TARGET_FS = 24  # Hz
BEAT_CORRELATION_THRESHOLD = 0.5

def is_good_quality(signal):
    if len(signal) != 10 * TARGET_FS:
        return False

    peaks, _ = find_peaks(signal, distance=int(TARGET_FS * 0.4), height=0)
    beats = []
    beat_window = int(0.7 * TARGET_FS)
    for peak in peaks:
        start = peak - beat_window // 2
        end = peak + beat_window // 2
        if start >= 0 and end <= len(signal):
            beats.append(signal[start:end])

    if len(beats) < 2:
        return False

    ref = beats[0]
    correlations = [np.corrcoef(ref, beat)[0, 1] for beat in beats[1:]]
    return np.mean(correlations) >= BEAT_CORRELATION_THRESHOLD if correlations else False
