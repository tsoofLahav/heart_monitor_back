
round_duration = 10
testing_mode = False


round_count = 0
round_peaks = []
last_sec = None
ave_gap = 0.7
round_signal = []


def reset_all():
    global round_count, round_peaks, last_sec, round_signal
    round_count = 0
    round_peaks = []
    last_sec = None
    round_signal = []


def add_to_round_peaks(peaks):
    global round_count, round_peaks
    round_peaks.extend([x + 10*round_count for x in peaks])


def add_to_round_signal(signal):
    global round_signal
    round_signal.extend([x for x in signal])
