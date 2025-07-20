import numpy as np
from flask import Flask, request, jsonify, send_file, make_response
import os
import logging

from video_edit import process_video_frames  # part 2: video -> signal
from filter_and_peaks import denoise_ppg  # part 4: filter + detect
from predict_model import predict_future_sequence  # part 6: prediction
import globals
from construct_raw_signal import connect_signals_with_gaps
from create_sound import generate_beep_track

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s", force=True)


def setup_video_route(app):
    @app.route('/process_video', methods=['POST'])
    def process_video():
        try:
            k = globals.round_duration  # local shorthand for clarity

            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩

            # ---------- Part 1: Get video ----------
            file = request.files.get('video')
            if not file:
                return jsonify({'error': 'No video file received.'}), 400

            video_path = './temp_video.mp4'
            file.save(video_path)
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise Exception("Invalid video file.")

            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩

            # ---------- Part 2: Video to signal ----------
            fps, intensities = process_video_frames(video_path, target_duration=k)
            if not intensities:
                raise Exception("No frames were processed.")

            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩

            globals.raw_buffer.append(intensities)

            if len(globals.raw_buffer) < 3:
                return jsonify({'loading': True})
            elif len(globals.raw_buffer) >= 3:
                # Keep only the latest 5 rounds
                if len(globals.raw_buffer) > 3:
                    globals.raw_buffer.pop(0)
                raw_signal = connect_signals_with_gaps(globals.raw_buffer[0], globals.raw_buffer[1], globals.raw_buffer[2])

            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩

            # ---------- Part 4: Filter + detect peaks + check signal quality ----------
            clean_signal, filtered_signal, not_reading, peaks_in_window = denoise_ppg(
                raw_signal, fps)

            if not_reading:
                globals.reset_all()
                return jsonify({'not_reading': True})

            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩

            # ---------- Part 5: Prepare model input from clean signal ----------
            # 2. Save last real peak
            last_real_peak = peaks_in_window[-1]

            # 3. Compute intervals and update global average
            intervals = [b - a for a, b in zip(peaks_in_window[:-1], peaks_in_window[1:])]
            globals.ave_gap = np.mean(intervals) if intervals else globals.ave_gap

            # 4. Adjust interval list to length 8 (add or trim)
            if len(intervals) < 8:
                intervals = [globals.ave_gap] * (8 - len(intervals)) + intervals
            else:
                intervals = intervals[-8:]

            # 5. Generate 8 peaks from intervals via cumulative sum
            generated_peaks = np.cumsum(intervals).tolist()

            # 6. Compute time shift for aligning prediction
            shift = last_real_peak - generated_peaks[-1]

            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩

            # ---------- Part 6: Predict future peaks ----------

            # 1. Prepare model input (8 intervals + 8 peaks)
            model_input = [val for i in range(8) for val in (intervals[i], generated_peaks[i])]

            # 2. Run prediction using loaded model
            predicted_peaks = predict_future_sequence(model_input)

            # 3. Shift prediction to align with actual time
            predicted_peaks = [t + shift for t in predicted_peaks]

            # 4. Extend if last predicted peak < 13.8 to fully cover 12–14s window
            while predicted_peaks[-1] < 14:
                predicted_peaks.append(predicted_peaks[-1] + globals.ave_gap)

            # 5. Keep only predicted peaks between 12–14s and shift to range 0–2s
            final_prediction = [t - 10.5 for t in predicted_peaks if 10.5 <= t <= 14]

            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩

            if globals.testing_mode:
                # Add latest prediction to the front of the buffer
                globals.prediction_buffer.insert(0, final_prediction)
                if len(globals.prediction_buffer) > 4:
                    globals.prediction_buffer.pop(4)

                # Use global function to build connected prediction
                connected_prediction = globals.construct_long_prediction()

                return jsonify({
                    'clean_signal': clean_signal.tolist(),
                    'filtered_signal': filtered_signal.tolist(),
                    'peaks_in_window': peaks_in_window,
                    'prediction': connected_prediction
                })

            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
            # --- Part 8: Handle edge gap correction ---
            # Check previous gap and adjust predictions if needed
            ave_interval = globals.ave_gap
            last_peak = final_prediction[-1]
            gap_to_end = 3.5 - last_peak

            if gap_to_end > ave_interval * 1:
                final_prediction.append(last_peak + ave_interval)

            if globals.last_gap is not None and len(final_prediction) > 0:
                first_candidate_time = ave_interval - globals.last_gap
                first_candidate_time = max(first_candidate_time, 0.0)

                if abs(final_prediction[0] - first_candidate_time) < 0.4 * ave_interval:
                    merged_time = (final_prediction[0] + first_candidate_time) / 2
                    final_prediction[0] = merged_time
                else:
                    final_prediction.insert(0, first_candidate_time)

            globals.last_gap = 3.5 - final_prediction[-1]
            # 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
            # ---------- Part 9: Save + send to frontend ----------
            globals.saved_predictions.append(final_prediction)
            audio_buffer = generate_beep_track(final_prediction)
            response = make_response(send_file(audio_buffer, mimetype="audio/wav", download_name="feedback.wav"))
            response.headers['X-BPM'] = str(60.0 / globals.ave_gap)
            return response

        except Exception as e:
            logging.exception("Unhandled exception:")
            globals.reset_all()
            return jsonify({'server_error': True}), 500

