import joblib
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.data.data_constants import BASELINE_FEATURES
from src.config import MODELS_DIR
from src.models.model import WinProbabilityModel

NN_DIR = MODELS_DIR/"neural_net"
NN_MODEL_FILENAME = "win_probability_network.pt"
NN_IMPUTER_FILENAME = "win_probability_imputer.joblib"
NN_SCALER_FILENAME = "win_probability_scaler.joblib"


def save_weights(model: nn.Module, imputer: SimpleImputer, scaler: StandardScaler) -> None:
    NN_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": BASELINE_FEATURES,
            "input_size": len(BASELINE_FEATURES),
        },
        NN_DIR / NN_MODEL_FILENAME,
    )
    joblib.dump(imputer, NN_DIR / NN_IMPUTER_FILENAME)
    joblib.dump(scaler, NN_DIR / NN_SCALER_FILENAME)


# Counterpart to save_weights - rebuilds the model architecture, loads its trained weights, and
# reloads the matching imputer/scaler so inference uses the exact preprocessing the model was
# trained on.
def load_artifacts(device: torch.device) -> tuple[nn.Module, SimpleImputer, StandardScaler]:
    checkpoint = torch.load(MODELS_DIR / NN_MODEL_FILENAME, map_location=device)

    model = WinProbabilityModel(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    imputer = joblib.load(MODELS_DIR / NN_IMPUTER_FILENAME)
    scaler = joblib.load(MODELS_DIR / NN_SCALER_FILENAME)

    return model, imputer, scaler
