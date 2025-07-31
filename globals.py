
round_duration = 10
testing_mode = False


round_count = 0
round_peaks = []
last_sec = None


def reset_all():
    global round_count, round_peaks, last_sec
    round_count = 0
    round_peaks = []
    last_sec = None


def add_to_round_peaks(peaks):
    round_peaks.extand([x + 10*round_count for x in peaks])
