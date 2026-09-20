import torch
from torch import nn

from src.models.config import DROPOUT_RATE, LAYER_ONE_WIDTH, LAYER_TWO_WIDTH


class WinProbabilityModel(nn.Module):
    # Initialise layers as well
    def __init__(self, input_size: int) -> None:
        super().__init__()

        # Sequential nn shape chosen as the data should always be the same shape and contain similar-ish numbers
        self.layers = nn.Sequential(
            # Takes in input features and expands it to more hidden features for accuracy
            nn.Linear(input_size, LAYER_ONE_WIDTH),

            # max(0,x). Introduces non-linearity as otherwise multiple layers would still be linear
            # Allows for relationships to be 'discovered'
            nn.ReLU(),

            # This is a dropout layer, which prevents overfitting to the dataset during training by setting 10% of
            # layer outputs to 0 for a training pass - not done during inference
            nn.Dropout(DROPOUT_RATE),

            nn.Linear(LAYER_ONE_WIDTH, LAYER_TWO_WIDTH),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(LAYER_TWO_WIDTH, 1),
        )

    # forward() defines the computation of the module
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)
