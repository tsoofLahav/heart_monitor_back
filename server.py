from flask import Flask, jsonify
from video_route import setup_video_route
import os
import globals
import random


app = Flask(__name__)

setup_video_route(app)
#app.register_blueprint(data_bp, url_prefix="/data")


# Health check route
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200


# Reset globals route.
@app.route('/end', methods=['POST'])
def end_session():
    try:
        # Step 1: Get real peaks from global
        real_peaks = globals.round_peaks.copy()
        duration = globals.round_count * 10

        # Step 2: Create noisy (fake) version with jitter
        noisy_peaks = []
        for p in real_peaks:
            jittered = round(p + random.uniform(-0.1, 0.1), 2)
            jittered = max(0.0, min(jittered, duration))
            noisy_peaks.append(jittered)

        # Step 3: Prepare response metadata
        metadata = {
            "peaks_count": len(real_peaks),
            "real_peaks": real_peaks,
            "fake_peaks": noisy_peaks,
            "duration": duration,
            "clean_signal": globals.round_signal
        }

        # Step 4: Reset global session state
        globals.reset_all()

        return jsonify(metadata), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port or default to 5000
    app.run(host="0.0.0.0", port=port)
