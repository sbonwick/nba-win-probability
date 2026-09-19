from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logging_utils import get_logger
from src.data.data_constants import BASELINE_FEATURES
from src.features.game_state import add_games_played
from src.config import PROCESSED_DIR, MODELS_DIR

TARGET_COLUMN = "home_won"
RANDOM_SEED = 42
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 5

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_split(data_type: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR/f"{data_type}.parquet")