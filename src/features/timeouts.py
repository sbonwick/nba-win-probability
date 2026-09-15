import pandas as pd

MODERN_TIMEOUTS = ["2017-18","2018-19","2019-20","2020-21","2021-22","2022-23",
                    "2023-24","2024-25","2025-26","2026-27"]

# Coach's challenges didn't exist before the 2019-20 season. Kept as a guard so
# a stray "Coach Challenge" subType in older data doesn't silently get treated
# as a real challenge (would indicate a data issue, not a rules issue).
CHALLENGE_ERAS = {"2019-20","2020-21","2021-22","2022-23","2023-24","2024-25",
                   "2025-26","2026-27"}


def add_timeout_features(df: pd.DataFrame, era: str) -> None:
    modern = era in MODERN_TIMEOUTS
    ot_floor = 2 if modern else 3

    df["home_timeouts_remaining"] = 7
    df["away_timeouts_remaining"] = 7
    home_timeouts = 7
    away_timeouts = 7
    period = None

    pending_challenge_side = None

    df.reset_index(drop=True, inplace=True)

    for index, row in df.iterrows():
        if row["period"] != period:
            period = row["period"]
            if period >= 5:
                home_timeouts = max(home_timeouts, ot_floor)
                away_timeouts = max(away_timeouts, ot_floor)

        if pending_challenge_side is not None and row["actionType"] == "Instant Replay":
            subtype = str(row.get("subType", "") or "")
            if "overturn" in subtype.lower():
                if pending_challenge_side == "home":
                    home_timeouts += 1
                else:
                    away_timeouts += 1
            pending_challenge_side = None

        if is_team_timeout(row):
            if row["is_home_event"] == 1:
                side = "home"
                home_timeouts -= 1
            elif row["is_home_event"] == 0:
                side = "away"
                away_timeouts -= 1
            else:
                side = None

            if side is not None and row["subType"] == "Coach Challenge":
                if era not in CHALLENGE_ERAS:
                    continue  # Ignore challenges in eras where they didn't exist.
                pending_challenge_side = side

        df.loc[index, "home_timeouts_remaining"] = home_timeouts
        df.loc[index, "away_timeouts_remaining"] = away_timeouts


def is_team_timeout(row: pd.Series) -> bool:
    return (
        row["actionType"] == "Timeout"
        and pd.notna(row["subType"])
        and row["subType"] in ["Regular", "Short", "Coach Challenge"]
    )


# def parse_timeouts_used(desc: str) -> int:
#     _,data = desc.split("(")
#     temp = ""
#     used = 0
#     for char in data:
#         if char.isnumeric():
#             temp += char
#             continue
#         if temp:
#             used += int(temp)
#             temp = ""
#     return used
