from src.utils.io import read_csv
from src.data.cache import boxScorePath, pbpPath
import pandas as pd

def merge_tables(game_id: str, season_type: str, season: str):
    box_score = read_csv(boxScorePath(season, game_id, season_type))
    pbp = read_csv(pbpPath(season, game_id, season_type))
    pbp.drop(["location","person_id"], axis=1, inplace=True)
    box_score_series = process_box_score(box_score)
    for col,value in box_score_series.items():
        pbp[col] = value

def process_box_score(box_score: pd.DataFrame) -> pd.Series:
    box_score.drop(["GAME_ID", "TEAM_ABBREVIATION", "PTS"], axis=1, inplace=True)
    home = box_score[box_score["IS_HOME"] == True]
    home = home.rename({
        "WINS": "home_wins",
        "LOSSES": "home_losses",
        "WON": "home_won",
        "TEAM_ID": "home_team_id",
    })
    away = box_score[box_score["IS_HOME"] == False]
    away = away.rename({
        "WINS": "away_wins",
        "LOSSES": "away_losses",
        "WON": "away_won",
        "TEAM_ID": "away_team_id",
    })
    return pd.Series(pd.concat([home, away], axis=1).values.flatten())
