import numpy as np
from flask import Flask, request, jsonify, send_file, make_response
import os
import logging

from video_edit import process_video_frames  # part 2: video -> signal
from filter_and_peaks import denoise_ppg, find_peaks  # part 4: filter + detect
import globals

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s", force=True)


def setup_video_route(app):
    @app.route('/process_video', methods=['POST'])
    def process_video():
        try:
            k = globals.round_duration  # local shorthand for clarity

            # ---------- Part 1: video process ----------
            file = request.files.get('video')
            if not file:
                return jsonify({'error': 'No video file received.'}), 400

            video_path = './temp_video.mp4'
            file.save(video_path)
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise Exception("Invalid video file.")

            fps, intensities = process_video_frames(video_path, target_duration=k)
            if not intensities:
                raise Exception("No frames were processed.")

            # ---------- Part 2: signal process ----------
            last_sec = globals.last_sec
            if last_sec is not None:
                intensities = np.concatenate([last_sec, intensities])
            clean_signal, filtered_signal, not_reading = denoise_ppg(intensities, fps)

            globals.last_sec = intensities[-fps:]

            if not_reading:
                return jsonify({'not_reading': True}), 200

            peaks_in_window = find_peaks(clean_signal, fps)
            final_peaks = [x for x in peaks_in_window if 0.5 <= x <= 10.5]
            globals.add_to_round_signal(clean_signal)

            # ---------- Part 3: return + store ----------
            globals.add_to_round_peaks(final_peaks)
            globals.round_count += 1

            if globals.testing_mode:
                return jsonify({
                    'clean_signal': clean_signal.tolist(),
                    'filtered_signal': filtered_signal.tolist(),
                    'peaks_in_window': peaks_in_window
                }), 200

            # ✅ Always return something
            return jsonify({'message': 'Processed successfully.'}), 200

        except Exception as e:
            logging.exception("Unhandled exception:")
            globals.reset_all()
            return jsonify({'server_error': True, 'error': str(e)}), 500
