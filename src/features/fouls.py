import re

import pandas as pd

COUNTABLE_FOULS = {
    "shooting",
    "personal",
    "loose ball",
    "personal take",
    "away from play",
    "inbound",
    "flagrant type 1",
    "flagrant type 2",
    "punching",
    "clear path foul",
}

PENALTY_TAG_PATTERN = re.compile(r"\(P\d+\.PN\)")


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

    df.reset_index(drop=True, inplace=True)

    for index, row in df.iterrows():
        if row["period"] != current_period:
            current_period = row["period"]
            current_home_fouls = 0
            current_away_fouls = 0
            home_in_penalty = False
            away_in_penalty = False

        if is_foul_event(row):
            side = get_foul_committing_side(row)

            if is_countable_team_foul(row):
                if side == 1:
                    current_home_fouls += 1
                elif side == 0:
                    current_away_fouls += 1

            if is_penalty_tagged(row):
                if side == 1:
                    home_in_penalty = True
                elif side == 0:
                    away_in_penalty = True

        df.loc[index, "home_team_fouls_period"] = current_home_fouls
        df.loc[index, "away_team_fouls_period"] = current_away_fouls
        df.loc[index, "home_in_penalty"] = home_in_penalty
        df.loc[index, "away_in_penalty"] = away_in_penalty


def is_countable_team_foul(row: pd.Series) -> bool:
    if pd.isna(row["subType"]):
        return False
    return row["subType"].strip().lower() in COUNTABLE_FOULS


def is_foul_event(row: pd.Series) -> bool:
    return pd.notna(row["actionType"]) and row["actionType"] == "Foul"


def is_penalty_tagged(row: pd.Series) -> bool:
    return pd.notna(row["description"]) and bool(
        PENALTY_TAG_PATTERN.search(row["description"])
    )


def get_foul_committing_side(row: pd.Series) -> float | None:
    return row["is_home_event"]