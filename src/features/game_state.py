from src.utils.io import read_csv
from src.data.cache import boxScorePath, pbpPath
import src.utils.time_utils as time_utils
import pandas as pd

def prepare_tables(game_id: str, season_type: str, season: str):
    box_score = read_csv(boxScorePath(season, game_id, season_type))
    pbp = read_csv(pbpPath(season, game_id, season_type))
    pbp.drop(["location"], axis=1, inplace=True)
    box_score_series = process_box_score(box_score)
    for col,value in box_score_series.items():
        pbp[col] = value
    pbp[season_type] = season_type

def get_event_side(df: pd.DataFrame) -> None:
    def event_side_helper(row: pd.Series) -> int | None:
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        if row["team_id"] == home_id or row["player_id"] == home_id:
            return 1
        elif row["team_id"] == away_id or row["player_id"] == away_id:
            return 0
        return None
    df["is_home_event"] = df.apply(event_side_helper, axis=1)

def add_time_features(df: pd.DataFrame) -> None:
    df["time_elapsed"] = df.apply(lambda row: time_utils.get_gametime_elapsed(row["period"], row["clock"]), axis=1)
    df["time_remaining"] = df.apply(lambda row: time_utils.get_gametime_remaining(row["period"], row["clock"]), axis=1)

def process_scores(df: pd.DataFrame) -> None:
    df["home_score"] = pd.to_numeric(df["scoreHome"],errors= "coerce").ffill().fillna(0)
    df["away_score"] = pd.to_numeric(df["scoreAway"],errors= "coerce").ffill().fillna(0)
    df["scoreDifferential"] = df["home_score"] - df["away_score"]

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
