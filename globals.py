# constants on runtime
round_duration = 3
testing_mode = False
local_mode = False

# video_route.
raw_buffer = []
peak_history = []
round_count = 0
prediction_buffer = []
ave_gap = None
last_gap = None

# data_route
session_id = None
saved_predictions = []

# models
mlp_model = None



def construct_long_prediction():
    if len(prediction_buffer) < 4:
        return []

    part1 = prediction_buffer[3]
    part2 = prediction_buffer[2]
    part3 = prediction_buffer[1]

    overlap_thresh = 0.1

    # Get last peak of part1 and first of part2
    if part1 and part2:
        if part2[0] <= overlap_thresh and (3.5 - part1[-1]) <= overlap_thresh:
            part2 = part2[1:]  # Skip first peak of part2

    if part2 and part3:
        if part3[0] <= overlap_thresh and (3.5 - part2[-1]) <= overlap_thresh:
            part3 = part3[1:]  # Skip first peak of part3

    # Return combined list (with shifted times)
    return (
        [t for t in part1] +
        [t + 3.5 for t in part2] +
        [t + 7 for t in part3]
    )



def reset_all():
    global raw_buffer, peak_history, round_count, prediction_buffer, ave_gap, saved_predictions
    raw_buffer = []
    peak_history = []
    round_count = 0
    prediction_buffer = []
    ave_gap = None
    saved_predictions = []
