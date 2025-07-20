import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
import globals
from unreadable_detection import is_good_quality


def butter_bandpass_filter(signal, fs, lowcut=0.8, highcut=3.0, order=6):
    """Applies a band-pass filter using second-order sections (SOS) for stability."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def regularize_signal(signal):
    """Normalize the signal to have mean 0 and std 1."""
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)  # Avoid division by zero


def denoise_ppg(raw_signal, fs):
    """
    Denoise PPG signal, check signal quality via beat correlation, and detect peaks.
    """
    raw_signal = np.array(raw_signal)

    # Step 1: Bandpass Filter
    filtered_signal = butter_bandpass_filter(raw_signal, fs)

    # Step 2: Regularization
    normalized_signal = regularize_signal(filtered_signal)

    # Step 3: New quality check
    if not is_good_quality(normalized_signal):
        return None, filtered_signal, True, []

    # Step 4: Peak detection
    all_positive = [x for x in normalized_signal if x > 0]
    avg_height = sum(all_positive) / len(all_positive) if all_positive else 0
    distance = (globals.ave_gap * 0.75 * fs) if globals.ave_gap else 0.5 * fs

    peaks, _ = find_peaks(normalized_signal, distance=distance, height=avg_height * 0.5)
    peak_times = (np.array(peaks) / fs).tolist()

    return normalized_signal, filtered_signal, False, peak_times



