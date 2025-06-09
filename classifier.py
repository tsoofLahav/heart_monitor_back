import torch
import torch.nn as nn
import globals
import requests
from io import BytesIO

url = "https://heart1monitor0storage.blob.core.windows.net/models/model2_classifier.pt"

# ---------- Define the model class ----------
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ---------- Load model (call once when session starts) ----------

def load_mlp_model():
    global url
    response = requests.get(url)
    response.raise_for_status()

    buffer = BytesIO(response.content)
    globals.mlp_model = SimpleMLP(globals.input_size)
    globals.mlp_model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
    globals.mlp_model.eval()
    print("✅ MLP model loaded successfully from URL")

# ---------- Use model to classify one segment ----------
def classify_signal_segment(segment):
    global mlp_model
    if globals.mlp_model is None:
        raise ValueError("Model not loaded. Call load_mlp_model() first.")

    with torch.no_grad():
        x = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # shape: (1, input_size)
        y_pred = globals.mlp_model(x).item()
        return 1 if y_pred >= 0.5 else 0
