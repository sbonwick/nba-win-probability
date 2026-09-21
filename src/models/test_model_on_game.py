import argparse

import joblib
import pandas as pd
import torch

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data.data_constants import BASELINE_FEATURES
from src.models.model import WinProbabilityModel
from src.models.preprocessing import transform_features
from src.models.persistence import NN_DIR,NN_MODEL_FILENAME,NN_IMPUTER_FILENAME,NN_SCALER_FILENAME
from src.models.torch_utils import get_device
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["training","validation","testing"],
        required=True,
        help = "Which parquet the processed game is in"
    )
    parser.add_argument(
        "--game-id",
        required=True
    )
    return parser.parse_args()

def load_game_frame(split_type: str, game_id: str) -> pd.DataFrame:
    path = PROCESSED_DIR/ f"{split_type}.parquet"
    frame = pd.read_parquet(path,filters=[("gameId","==",game_id)])

    if frame.empty:
        raise ValueError(f"gameId={game_id} was not found in {path}")

    frame["gameId"] = frame["gameId"].astype("string")
    frame = frame.sort_values(["period","time_elapsed"],kind = "stable").reset_index(drop=True)
    return frame

def load_model_and_preprocess(device: torch.device) -> tuple[WinProbabilityModel,object,object]:
    checkpoint = torch.load(NN_DIR/NN_MODEL_FILENAME,map_location=device,weights_only=True)
    model = WinProbabilityModel(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    imputer = joblib.load(NN_DIR/NN_IMPUTER_FILENAME)
    scaler = joblib.load(NN_DIR/NN_SCALER_FILENAME)
    return model,imputer,scaler

def predict_game_probabilities(model:WinProbabilityModel,frame:pd.DataFrame,imputer,scaler,device:torch.device) -> pd.DataFrame:
    features = transform_features(frame,imputer,scaler)
    feature_tensor = torch.as_tensor(features,dtype=torch.float32,device=device)

    with torch.inference_mode():
        outcomes = model(feature_tensor)
        probabilities = torch.sigmoid(outcomes).squeeze(1).cpu().numpy()

    result = frame.copy()
    result["home_win_probability"] = probabilities
    return result

def save_predictions(frame:pd.DataFrame,game_id:str) -> None:
    output_dir = REPORTS_DIR/ "game_predictions"
    output_dir.mkdir(parents=True,exist_ok=True)
    output_path = output_dir/ f"{game_id}_probabilities.csv"
    frame.to_csv(output_path,index=False)
    logger.info("Saved %d game state predictions to %s", len(frame),output_path)

def main() -> None:
    args = parse_arguments()
    device = get_device()
    logger.info("predicting gameId=%s on device=%s",args.game_id,device)
    frame = load_game_frame(args.split,args.game_id)
    model,imputer,scaler = load_model_and_preprocess(device)
    predicted_frame = predict_game_probabilities(model,frame,imputer,scaler,device)
    save_predictions(predicted_frame,args.game_id)

if __name__ == "__main__":
    main()