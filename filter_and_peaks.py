import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

import globals
from classifier import classify_signal_segment
from reconstruction import reconstruct_missing


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
    Denoise PPG signal, do simple quality check per segment, reconstruct if needed, detect peaks.
    """
    raw_signal = np.array(raw_signal)

    # Step 1: Bandpass Filter
    filtered_signal = butter_bandpass_filter(raw_signal, fs)

    # Step 2: Regularization
    normalized_signal = regularize_signal(filtered_signal)

    # Step 3: Segment-wise quality check
    window_size = int(2 * fs)
    num_windows = len(normalized_signal) // window_size
    bad_windows = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        segment = normalized_signal[start:end]

        std = np.std(segment)
        amp = np.max(segment) - np.min(segment)
        avg_height = np.mean([x for x in segment if x > 0]) if any(x > 0 for x in segment) else 0

        if std < 0.05 or amp < 0.5 or avg_height < 0.1:
            bad_windows.append(i)

    # Step 4: Decision logic
    if len(bad_windows) >= 2:
        return None, filtered_signal, True, []

    if len(bad_windows) == 1:
        bad_window = bad_windows[0]
        bad_start = bad_window * window_size
        bad_end = bad_start + window_size

        # Zero out bad part
        corrupted_signal = normalized_signal.copy()
        corrupted_signal[bad_start:bad_end] = 0.0

        # Reconstruct
        reconstructed = reconstruct_missing(corrupted_signal)

        # Replace only that window
        normalized_signal[bad_start:bad_end] = reconstructed[bad_start:bad_end]

    # Step 5: Peak detection
    all_positive = [x for x in normalized_signal if x > 0]
    avg_height = sum(all_positive) / len(all_positive) if all_positive else 0
    distance = (globals.ave_gap * 0.75 * fs) if globals.ave_gap else 0.5 * fs

    peaks, _ = find_peaks(normalized_signal, distance=distance, height=avg_height * 0.5)
    peak_times = (np.array(peaks) / fs).tolist()

    return normalized_signal, filtered_signal, False, peak_times

