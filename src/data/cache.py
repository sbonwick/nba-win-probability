from pathlib import Path
from src.config import RAW_DIR

def normalize_game_type(game_type: str) -> str:
    return game_type.strip().lower().replace(" ", "_")

def gameIDPath(season:str,game_type:str) -> Path:
    return Path(RAW_DIR/"game_ids"/f"{season}_{normalize_game_type(game_type)}.csv")

def pbpPath(season:str,game_id:str,game_type:str):
    return Path(RAW_DIR/"pbp"/season/normalize_game_type(game_type)/f"{game_id}.csv")

def boxScorePath(season:str,game_id:str,game_type:str):
    return Path(RAW_DIR/"box_scores"/season/normalize_game_type(game_type)/f"{game_id}.csv")

def failurePath(season:str):
    return Path(RAW_DIR/"pbp"/"failures"/f"{season}_failures.csv")

def gameIDFileexists(season:str,season_type:str)->bool:
    return gameIDPath(season,season_type).exists()

def pbpFileExists(season:str,game_id:str,season_type:str) -> bool:
    return pbpPath(season,game_id,season_type).exists()

def isCached(path:Path) -> bool:
    return path.is_file()