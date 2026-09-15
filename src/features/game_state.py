from src.utils.io import read_csv
from src.data.cache import boxScorePath, pbpPath
import src.utils.time_utils as time_utils
import pandas as pd
from src.features.possession import add_possession
from src.features.fouls import add_foul_features
from src.features.timeouts import add_timeout_features

def prepare_tables(game_id: str, season_type: str, season: str) -> pd.DataFrame:
    box_score = read_csv(boxScorePath(season, game_id, season_type))
    pbp = read_csv(pbpPath(season, game_id, season_type))
    pbp.drop(["location"], axis=1, inplace=True)
    box_score_series = process_box_score(box_score)
    for col,value in box_score_series.items():
        pbp[col] = value
    get_event_side(pbp)
    pbp["playoffs"] = 1 if season_type == "Playoffs" else 0
    add_time_features(pbp)
    process_scores(pbp)
    add_foul_features(pbp)
    add_possession(pbp)
    add_timeout_features(pbp,season)
    pbp.drop(["actionNumber","clock","teamId","personId","actionType","subType","description","is_home_event","home_team_id","away_team_id"], axis=1, inplace=True)
    home_games = pbp["home_wins"] + pbp["home_losses"]
    away_games = pbp["away_wins"] + pbp["away_losses"]

    pbp["home_win_pct"] = (pbp["home_wins"] / home_games).where(home_games > 0, 0.5)
    pbp["away_win_pct"] = (pbp["away_wins"] / away_games).where(away_games > 0, 0.5)

    pbp["win_pct_diff"] = pbp["home_win_pct"] - pbp["away_win_pct"]
    pbp["is_overtime"] = (pbp["period"] > 4)
    pbp = pbp.astype({
    "period":                      "int8",
    "scoreHome":                   "int16",
    "scoreAway":                   "int16",
    "home_wins":                   "int16",
    "home_losses":                 "int16",
    "home_won":                    "bool",
    "away_wins":                   "int16",
    "away_losses":                 "int16",
    "playoffs":                    "bool",
    "time_elapsed":                "int16",
    "time_remaining":              "int16",
    "scoreDifferential":           "int16",
    "home_team_fouls_period":      "int8",
    "away_team_fouls_period":      "int8",
    "home_in_penalty":             "bool",
    "away_in_penalty":             "bool",
    "home_timeouts_remaining":     "int8",
    "away_timeouts_remaining":     "int8",
    "gameId":                      "str",
    "home_win_pct":                "float32",
    "away_win_pct":                "float32",
    "win_pct_diff":                "float32",
    "is_overtime":                 "bool",
    })
    pbp["home_possession"] = pbp["home_possession"].astype("boolean")  # nullable bool
    return pbp


def get_event_side(df: pd.DataFrame) -> None:
    def event_side_helper(row: pd.Series) -> int | None:
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        if row["teamId"] == home_id or row["personId"] == home_id:
            return 1
        elif row["teamId"] == away_id or row["personId"] == away_id:
            return 0
        return None
    df["is_home_event"] = df.apply(event_side_helper, axis=1)

def add_time_features(df: pd.DataFrame) -> None:
    df["time_elapsed"] = df.apply(lambda row: time_utils.get_gametime_elapsed(row["period"], row["clock"]), axis=1)
    df["time_remaining"] = df.apply(lambda row: time_utils.get_gametime_remaining(row["period"], row["clock"]), axis=1)

def process_scores(df: pd.DataFrame) -> None:
    df["scoreHome"] = pd.to_numeric(df["scoreHome"],errors= "coerce").ffill().fillna(0)
    df["scoreAway"] = pd.to_numeric(df["scoreAway"],errors= "coerce").ffill().fillna(0)
    df["scoreDifferential"] = df["scoreHome"] - df["scoreAway"]

def process_box_score(box_score: pd.DataFrame) -> pd.Series:
    box_score.drop(["GAME_ID", "TEAM_ABBREVIATION", "PTS"], axis=1, inplace=True)

    home = box_score[box_score["IS_HOME"] == True].iloc[0].rename({
        "WINS": "home_wins",
        "LOSSES": "home_losses",
        "WON": "home_won",
        "TEAM_ID": "home_team_id",
        "IS_HOME": "home_is_home",
    })

    away = box_score[box_score["IS_HOME"] == False].iloc[0].rename({
        "WINS": "away_wins",
        "LOSSES": "away_losses",
        "WON": "away_won",
        "TEAM_ID": "away_team_id",
        "IS_HOME": "away_is_home",
    })

    frame = pd.concat([home, away])
    frame.drop(["home_is_home", "away_is_home", "away_won"], inplace=True)
    frame["home_won"] = int(frame["home_won"])

    return frame
