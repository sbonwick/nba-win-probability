from pathlib import Path
from src.config import RAW_DIR,INTERIM_DIR

def normalize_game_type(game_type: str) -> str:
    return game_type.strip().lower().replace(" ", "_")

def gameIDPath(season:str,game_type:str) -> Path:
    return Path(RAW_DIR/"game_ids"/f"{season}_{normalize_game_type(game_type)}.csv")

def pbpPath(season:str,game_id:str,game_type:str):
    return Path(RAW_DIR/"pbp"/season/normalize_game_type(game_type)/f"{game_id}.csv")

def boxScorePath(season:str,game_id:str,game_type:str):
    return Path(RAW_DIR/"box_scores"/season/normalize_game_type(game_type)/f"{game_id}.csv")

def pbpFailurePath(season:str, game_type:str):
    return Path(RAW_DIR/"failures"/"pbp"/season/normalize_game_type(game_type)/f"{season}_failures.csv")

def boxScoreFailurePath(season:str, game_type:str):
    return Path(RAW_DIR/"failures"/"box_scores"/season/normalize_game_type(game_type)/f"{season}_failures.csv")

def interimDataFramePath(season:str,game_id:str,game_type:str):
    return Path(INTERIM_DIR/season/normalize_game_type(game_type)/f"{game_id}.csv")

def interimDataFrameFailurePath(season:str,game_type:str):
    return Path(INTERIM_DIR/"failures"/season/normalize_game_type(game_type)/f"{season}_failures.csv")

def gameIDFileexists(season:str,season_type:str)->bool:
    return gameIDPath(season,season_type).exists()

def pbpFileExists(season:str,game_id:str,season_type:str) -> bool:
    return pbpPath(season,game_id,season_type).exists()

def interimDataFrameExist(season:str,game_id:str,season_type:str) -> bool:
    return interimDataFramePath(season,game_id,season_type).exists()

def isCached(path:Path) -> bool:
    return path.is_file()