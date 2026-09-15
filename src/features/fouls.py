import pandas as pd

COUNTABLE_FOULS = [
    "Shooting",
    "Personal",
    "Loose Ball",
    "Personal Take",
    "Offensive Charge",
    "Away From Play",
    "Inbound",
]

def add_foul_features(df: pd.DataFrame) -> None:
    df["home_team_fouls_period"] = 0
    df["away_team_fouls_period"] = 0
    df["home_in_penalty"] = False
    df["away_in_penalty"] = False

    current_period = None
    current_home_fouls = 0
    current_away_fouls = 0
    home_in_penalty = False
    away_in_penalty = False

    df.reset_index(drop=True,inplace=True)

    for index, row in df.iterrows():
        foul_limit = 5 if row["period"] <= 4 else 4

        if row["period"] != current_period:
            current_period = row["period"]
            current_home_fouls = 0
            current_away_fouls = 0
            home_in_penalty = False
            away_in_penalty = False

        elif is_foul_event(row) and is_countable_team_foul(row):
            if get_foul_committing_side(row) == 1:
                current_home_fouls += 1
                if current_home_fouls >= foul_limit:
                    home_in_penalty = True

            elif get_foul_committing_side(row) == 0:
                current_away_fouls += 1
                if current_away_fouls >= foul_limit:
                    away_in_penalty = True

        df.loc[index, "home_team_fouls_period"] = current_home_fouls
        df.loc[index, "away_team_fouls_period"] = current_away_fouls
        df.loc[index, "home_in_penalty"] = home_in_penalty
        df.loc[index, "away_in_penalty"] = away_in_penalty


def is_countable_team_foul(row: pd.Series) -> bool:
    return pd.notna(row["subType"]) and row["subType"] in COUNTABLE_FOULS


def is_foul_event(row: pd.Series) -> bool:
    return pd.notna(row["actionType"]) and row["actionType"] == "Foul"


def get_foul_committing_side(row: pd.Series) -> int | None:
    return row["is_home_event"]