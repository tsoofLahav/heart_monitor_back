import torch
import torch.nn as nn
import globals
import requests
from io import BytesIO

# Updated URL
url = "https://myheartappstorage.blob.core.windows.net/models/mlp_model.pt"

# Updated model structure to MLP
class MLP(nn.Module):
    def __init__(self, input_dim=2, seq_len=8, output_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # (B, 6, 2) -> (B, 12)
            nn.Linear(input_dim * seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)  # (B, output_size)

def load_predictor_model():
    model = MLP(input_dim=2, seq_len=8, output_size=8)

    if getattr(globals, "local_mode", False):
        model.load_state_dict(torch.load("deep_lstm_model.pt", map_location='cpu'))
        print("✅ Predictor model loaded from local file.")
    else:
        response = requests.get(url)
        response.raise_for_status()
        buffer = BytesIO(response.content)
        model.load_state_dict(torch.load(buffer, map_location='cpu'))
        print("✅ Predictor model loaded from URL.")

    model.eval()
    globals.predictor_model = model

def predict_future_sequence(input_intervals_and_peaks):

    if len(input_intervals_and_peaks) < 16:
        raise ValueError("Expected 16 input values (8 intervals + 8 peaks)")

    sequence = torch.tensor(input_intervals_and_peaks[:16], dtype=torch.float32).reshape(1, 8, 2)

    model = globals.predictor_model
    with torch.no_grad():
        predicted_peaks = model(sequence)

    return predicted_peaks.squeeze(0).tolist()

