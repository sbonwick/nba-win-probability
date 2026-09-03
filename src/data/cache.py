from pathlib import Path
from src.config import RAW_DIR

def gameIDPath(season:str,game_type:str) -> Path:
    return Path(RAW_DIR/"game_ids"/f"{season}_{game_type}.csv")

def pbpPath(season:str,game_id:str):
    return Path(RAW_DIR/"pbp"/season/f"{game_id}.json")

def failurePath(season:str):
    return Path(RAW_DIR/"pbp"/"failures"/f"{season}_failures.csv")

def gameIDFileexists(season:str,season_type:str)->bool:
    return gameIDPath(season,season_type).exists()

def pbpFileExists(season:str,game_id:str) -> bool:
    return pbpPath(season,game_id).exists()

def isCached(path:Path) -> bool:
    return path.is_file()