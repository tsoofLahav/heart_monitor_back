from flask import Flask, jsonify
from video_route import setup_video_route
import os
import globals
import random
from create_sound_and_vid import generate_heartbeat_video


app = Flask(__name__)

setup_video_route(app)
#app.register_blueprint(data_bp, url_prefix="/data")


# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

# Reset globals route.
@app.route('/end', methods=['POST'])
def end_session():
    try:
        # Step 1: Get real peaks from global
        real_peaks = globals.round_peaks.copy()
        duration = globals.round_count * 10

        # Step 2: Create noisy version (same length, slight jitter)
        noisy_peaks = []
        for p in real_peaks:
            jittered = round(p + random.uniform(-0.1, 0.1), 2)
            jittered = max(0.0, min(jittered, duration))
            noisy_peaks.append(jittered)

        # Step 3: Generate both videos
        os.makedirs("static", exist_ok=True)
        real_path = "static/real_heartbeat.mp4"
        fake_path = "static/fake_heartbeat.mp4"

        generate_heartbeat_video(real_peaks, duration, output_path=real_path)
        generate_heartbeat_video(noisy_peaks, duration, output_path=fake_path)

        # Step 4: Prepare and send metadata
        metadata = {
            "peaks_count": len(real_peaks),
            "real_video_url": "/static/real_heartbeat.mp4",
            "fake_video_url": "/static/fake_heartbeat.mp4"
        }

        # Step 5: Reset global session state
        globals.reset_all()

        return jsonify(metadata), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port or default to 5000
    app.run(host="0.0.0.0", port=port)
