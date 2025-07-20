from pydub import AudioSegment
from io import BytesIO

def generate_beep_track(peaks, total_duration=3.5, beep_path="beep.wav"):
    beep = AudioSegment.from_wav(beep_path)
    track = AudioSegment.silent(duration=int(total_duration * 1000))

    for peak in peaks:
        start_ms = int(peak * 1000)
        if start_ms + len(beep) <= len(track):
            track = track.overlay(beep, position=start_ms)

    buffer = BytesIO()
    track.export(buffer, format="wav")
    buffer.seek(0)
    return buffer  # return to main page
