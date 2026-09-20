import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.config import LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE
from src.models.evaluation import predict_probability, calculate_metrics
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def setup_loss_optimiser(model: nn.Module) -> tuple[nn.BCEWithLogitsLoss, torch.optim.Optimizer]:

    # This is chosen as it is more stable than applying a sigmoid then binary cross-entropy loss function
    loss_function = nn.BCEWithLogitsLoss()

    # This was chosen over SGD as it adapts parameters with larger gradients more than those with smaller ones
    # Leads to quicker results and works well with standardised inputs
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    return loss_function, optimiser


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for features, targets in loader:
        # Converts inputs and targets to correct device - GPU/CPU
        features = features.to(device)
        targets = targets.to(device)
        # This is the standard, key loop in pytorch nns
        # Passes the data through the current model
        outputs = model(features)
        loss = loss_function(outputs, targets)
        # Zeroes the gradients as these accumulate over each epoch
        optimiser.zero_grad()
        # Performs backpropagation
        loss.backward()
        # Updates the model's parameters using gradient descent
        optimiser.step()

        total_loss += loss.item() * len(features)
        total_rows += len(features)

    return total_loss / total_rows


def fit_model(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    validation_targets: np.ndarray,
    device: torch.device,
) -> nn.Module:
    # Model must live on the training device before its parameters are handed to the optimiser
    model.to(device)
    loss_function, optimiser = setup_loss_optimiser(model)

    best_state = None
    best_validation_loss = float("inf")

    # Used to prevent getting stuck in local minimums
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, training_loader, loss_function, optimiser, device)
        validation_probabilities = predict_probability(model, validation_loader, device)
        validation_log_loss = calculate_metrics(validation_targets, validation_probabilities)["log_loss"]
        logger.info(
            "Epoch %d: training loss = %.4f, validation_log_loss = %.4f",
            epoch, train_loss, validation_log_loss,
        )

        if validation_log_loss < best_validation_loss:
            best_validation_loss = validation_log_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model
