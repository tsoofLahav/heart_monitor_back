from flask import Flask, jsonify
from video_route import setup_video_route
import os
from predict_model import load_predictor_model
import globals


app = Flask(__name__)

setup_video_route(app)
#app.register_blueprint(data_bp, url_prefix="/data")


# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

# Load models route (for local testing)
@app.route('/load_models', methods=['GET'])
def load_models():
    globals.reset_all()
    load_predictor_model()
    return jsonify({"status": "Models loaded"}), 200

# Reset globals route
@app.route('/end', methods=['POST'])
def end_session():
    return jsonify({"predictions": globals.saved_predictions}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port or default to 5000
    app.run(host="0.0.0.0", port=port)
