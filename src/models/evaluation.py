import numpy as np
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def predict_probability(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    # This ignores the dropout layer as the model is no longer being trained and having its values altered
    model.eval()
    probabilities: list[np.ndarray] = []

    # This is used for making predictions - this mode turns off gradient tracking among other things
    # to speed up the forward method
    with torch.inference_mode():
        for batch in loader:
            # batch[0] is always the feature tensor, whether or not a target tensor is also present -
            # this lets the same function serve training/validation loaders (features, targets) and
            # inference-only loaders (features,) where no ground truth outcome exists yet
            features = batch[0]
            outputs = model(features.to(device))
            # The sigmoid maps the output to a number between 0 and 1
            # Moves it to CPU memory (if necessary), converts tensor to numpy array and 'flattens' it
            batch_probabilities = torch.sigmoid(outputs).cpu().numpy().ravel()
            probabilities.append(batch_probabilities)

    return np.concatenate(probabilities)


# Calculates 'quality' of model in iteration through certain metrics
#
# Log loss - How good the predictions are and penalises a lot for predictions that are confident and wrong
# Brier score - Mean squared difference between each probability and the actual outcome
# ROC-AUC - Measures whether the win outputs correlate with the probabilities - agreement, not accuracy
def calculate_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    # Prevents exact 0 or 1 from causing issues
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "log_loss": log_loss(targets, clipped),
        "brier_score": brier_score_loss(targets, clipped),
        "roc_auc": roc_auc_score(targets, clipped),
    }
