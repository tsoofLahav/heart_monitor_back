from moviepy.editor import VideoClip, AudioClip
import numpy as np
import cv2
import os


def generate_heartbeat_video(peaks, duration, output_path="heartbeat_video.mp4"):
    """
    Generates a heartbeat monitor-style video with scrolling ECG line and synced beep sound.

    Args:
        peaks (list of float): Times (in seconds) of heartbeat peaks.
        duration (float): Total duration of the video.
        output_path (str): Path to save the generated video.
    """
    width, height = 600, 400
    fps = 24
    signal_length = width  # one value per horizontal pixel

    # Generate a single ECG-like pulse shape
    def generate_ecg_pulse(length=30):
        t = np.linspace(0, 1, length)
        pulse = np.exp(-((t - 0.2) * 20) ** 2) * 1.0 \
                - np.exp(-((t - 0.5) * 30) ** 2) * 0.5 \
                + np.exp(-((t - 0.7) * 20) ** 2) * 0.3
        return pulse

    pulse_shape = generate_ecg_pulse()
    signal_buffer = np.zeros(signal_length)

    def make_frame(t):
        nonlocal signal_buffer
        frame = np.zeros((height, width, 3), dtype=np.uint8)  # black background

        # If a peak just occurred, insert pulse at the end
        if any(abs(t - p) < 1 / fps for p in peaks):
            insert_pos = -len(pulse_shape)
            signal_buffer[insert_pos:] += pulse_shape

        # Shift signal left
        signal_buffer = np.roll(signal_buffer, -1)
        signal_buffer[-1] = 0  # Clear the newly inserted position

        # Draw ECG line
        ecg_y = height // 2 - (signal_buffer * 100).astype(int)  # scale & center
        for x in range(1, width):
            y1, y2 = ecg_y[x - 1], ecg_y[x]
            if 0 <= y1 < height and 0 <= y2 < height:
                cv2.line(frame, (x - 1, y1), (x, y2), (0, 255, 0), 2)

        return frame

    video = VideoClip(make_frame, duration=duration).set_fps(fps)

    # Audio: sine beep at each peak
    def sine_beep(frequency=880, length=0.1, rate=44100):
        t = np.linspace(0, length, int(rate * length))
        return 0.5 * np.sin(2 * np.pi * frequency * t)

    def beep_audio(t):
        frame_size = int(44100 / 10)
        if any(abs(t - p) < 0.05 for p in peaks):
            return sine_beep()[:frame_size]
        return np.zeros(frame_size)

    audio = AudioClip(beep_audio, duration=duration, fps=44100)
    video_with_audio = video.set_audio(audio)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
