import numpy as np
from scipy.interpolate import CubicSpline

def connect_signals_with_gaps(part1, part2, part3):
    """
    Connects 3 raw signal parts with 0.25s (6-frame) interpolated gaps.
    Each part must be length 72 (3 sec at 24 fps).
    Output: full signal of 228 frames (9.5 sec).
    """
    assert len(part1) == len(part2) == len(part3) == 72, "Each part must be 72 frames."

    def interpolate_gap(start_val, end_val, num_frames=12):
        return CubicSpline([0, num_frames + 1], [start_val, end_val])(np.arange(1, num_frames + 1))

    # Interpolate between part1[-1] and part2[0]
    gap1 = interpolate_gap(part1[-1], part2[0])

    # Interpolate between part2[-1] and part3[0]
    gap2 = interpolate_gap(part2[-1], part3[0])

    # Concatenate: part1 + gap1 + part2 + gap2 + part3
    full_signal = np.concatenate([part1, gap1, part2, gap2, part3])
    assert len(full_signal) == 240, f"Expected 228 frames, got {len(full_signal)}"

    return full_signal
