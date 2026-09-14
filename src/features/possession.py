import re
import pandas as pd


def add_possession(df: pd.DataFrame) -> None:
    df.reset_index(drop=True, inplace=True)
    df["home_possession"] = pd.NA

    home_possession = None

    for index, row in df.iterrows():
        next_row = df.iloc[index + 1] if index + 1 < len(df) else None

        action_type = row["actionType"]

        if action_type == "Jump Ball":
            home_possession = int(bool(row["is_home_event"]))

        elif action_type == "Rebound":
            home_possession = int(bool(row["is_home_event"]))

        elif action_type == "Turnover":
            if pd.notna(row["is_home_event"]):
                home_possession = 0 if bool(row["is_home_event"]) else 1

        elif action_type == "Made Shot":
            result = possession_after_made_shot(row, next_row)
            if result is not None:
                home_possession = result

        elif action_type == "Free Throw":
            result = possession_after_free_throw(row)
            if result is not None:
                home_possession = result

        elif home_possession is None:
            inferred = infer_initial_possession(row)
            if inferred is not None:
                home_possession = inferred

        df.loc[index, "home_possession"] = home_possession


def infer_initial_possession(row: pd.Series) -> int | None:
    action_type = row["actionType"]

    if action_type in ["Made Shot", "Missed Shot", "Free Throw", "Turnover", "Rebound"]:
        if pd.notna(row["is_home_event"]):
            return int(bool(row["is_home_event"]))

    if action_type == "Foul":
        if row["subType"] == "Shooting" and pd.notna(row["is_home_event"]):
            return 0 if bool(row["is_home_event"]) else 1

    return None


def possession_after_made_shot(row: pd.Series, next_row: pd.Series | None) -> int | None:
    if pd.isna(row["is_home_event"]):
        return None

    scoring_team_home = int(bool(row["is_home_event"]))

    if is_and_one_sequence(row, next_row):
        return None

    return 0 if scoring_team_home == 1 else 1


def is_and_one_sequence(row: pd.Series, next_row: pd.Series | None) -> bool:
    if next_row is None:
        return False

    if next_row["actionType"] != "Foul":
        return False

    if next_row["subType"] != "Shooting":
        return False

    if pd.isna(next_row["is_home_event"]) or pd.isna(row["is_home_event"]):
        return False

    return bool(next_row["is_home_event"]) != bool(row["is_home_event"])


def possession_after_free_throw(row: pd.Series) -> int | None:
    sub_type = row["subType"]
    description = row["description"]

    if pd.isna(sub_type) or pd.isna(description):
        return None

    if "Technical" in sub_type or "Flagrant" in sub_type:
        return None

    if description.startswith("MISS"):
        return None

    match = re.search(r"(\d+) of (\d+)", sub_type)
    if match is None:
        return None

    shot_num = int(match.group(1))
    total_shots = int(match.group(2))

    if shot_num != total_shots:
        return None

    if pd.isna(row["is_home_event"]):
        return None

    shooting_team_home = int(bool(row["is_home_event"]))
    return 0 if shooting_team_home == 1 else 1