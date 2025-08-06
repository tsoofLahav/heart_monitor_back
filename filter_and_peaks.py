import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks as scipy_find_peaks
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
    Denoise PPG signal, check signal quality via beat correlation.
    Returns: (normalized_signal, filtered_signal, not_reading)
    """
    raw_signal = np.array(raw_signal)

    # Step 1: Bandpass Filter
    filtered_signal = butter_bandpass_filter(raw_signal, fs)

    # Step 2: Regularization
    normalized_signal = regularize_signal(filtered_signal)

    # Step 3: Signal quality check
    #if not is_good_quality(normalized_signal):
    #    return None, filtered_signal, True

    return normalized_signal, filtered_signal, False


def find_peaks(signal, fs=None):
    """
    Find peaks in the signal. If fs is provided, uses dynamic distance.
    Returns list of peak times in seconds.
    """
    signal = np.array(signal)

    all_positive = signal[signal > 0]
    avg_height = np.mean(all_positive) if len(all_positive) > 0 else 0
    distance = int(fs*0.25)

    peaks, _ = scipy_find_peaks(signal, distance=distance, height=avg_height * 0.5)

    if fs:
        return (peaks / fs).tolist()
    else:
        return peaks.tolist()



