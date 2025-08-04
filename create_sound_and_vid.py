import numpy as np
from PIL import Image, ImageDraw
import os
import subprocess
from scipy.io.wavfile import write as write_wav

def generate_ecg_frame(signal, width=600, height=400):
    img = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(img)
    mid_y = height // 2
    scale = 100

    for x in range(1, len(signal)):
        y1 = mid_y - int(signal[x - 1] * scale)
        y2 = mid_y - int(signal[x] * scale)
        draw.line([(x - 1, y1), (x, y2)], fill='green', width=2)

    return img

def generate_beep_wave(peaks, duration, rate=44100):
    t = np.linspace(0, duration, int(rate * duration))
    signal = np.zeros_like(t)
    for p in peaks:
        idx = int(p * rate)
        if idx + 500 < len(signal):  # 500 samples ~ 0.01s
            signal[idx:idx+500] += 0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, 0.01, 500))
    return (signal * 32767).astype(np.int16)

def generate_heartbeat_video_safe(peaks, duration, output_path="heartbeat_video.mp4"):
    fps = 24
    frame_count = int(duration * fps)
    signal = np.zeros(600)
    pulse = np.exp(-((np.linspace(0, 1, 30) - 0.5) * 30) ** 2)

    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(frame_count):
        t = i / fps
        if any(abs(t - p) < 1 / fps for p in peaks):
            signal[-len(pulse):] += pulse

        signal = np.roll(signal, -1)
        signal[-1] = 0
        img = generate_ecg_frame(signal)
        img.save(f"{frames_dir}/frame_{i:04d}.png")

    # Generate beep audio
    audio = generate_beep_wave(peaks, duration)
    write_wav("beep.wav", 44100, audio)

    # Create video using ffmpeg
    subprocess.run([
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/frame_%04d.png",
        "-i", "beep.wav",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path
    ])

    # Cleanup
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))
    os.rmdir(frames_dir)
    os.remove("beep.wav")
