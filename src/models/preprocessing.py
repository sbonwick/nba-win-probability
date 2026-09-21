import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_constants import BASELINE_FEATURES
from src.config import PROCESSED_DIR
from src.features.game_state import add_games_played, add_score_time_relationship


def load_split(data_type: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{data_type}.parquet")


# This function has 2 purposes, which are the two objects returned
def fit_preprocesser(training_frame: pd.DataFrame) -> tuple[SimpleImputer, StandardScaler]:

    # Firstly, it will fill in any missing values: currently there should be no missing values whatsoever
    imputer = SimpleImputer(strategy="median")
    # It also scales input features - since time remaining is such a large number at the start (2880) and
    # home_in_penalty is 0 to 1, this will stop numbers which are naturally larger dominating the optimisation
    scaler = StandardScaler()
    training_values = imputer.fit_transform(training_frame[BASELINE_FEATURES])
    scaler.fit(training_values)
    return imputer, scaler


def transform_features(frame: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    values = imputer.transform(frame[BASELINE_FEATURES])
    return scaler.transform(values).astype(np.float32)


# This provides the loader for the model so the data is supplied as tensors in batches
def make_loader(features: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:

    # Shape is (num_rows, num_features)
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)

    # Changes shape from (rows,) to (rows, 1) so it matches one target logit per row
    target_tensor = target_tensor.unsqueeze(1)

    # Pairs the data together into game state and match outcome essentially
    dataset = TensorDataset(feature_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Used at real prediction time, where there is no known outcome to pair with the features.
# Yields single-element batches (features,) rather than (features, targets).
def make_inference_loader(features: np.ndarray, batch_size: int) -> DataLoader:
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(feature_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "home_games_played" not in frame.columns:
        add_games_played(frame)
    if "score_time_relationship" not in frame.columns:
        add_score_time_relationship(frame)
    return frame
    