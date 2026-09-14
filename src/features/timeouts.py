import pandas as pd
MODERN_TIMEOUTS = ["2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26","2026-27"]


def add_timeouts(df: pd.DataFrame, era: str) -> None:
    modern = era in MODERN_TIMEOUTS
    df["home_timeouts_remaining"] = 7
    df["away_timeouts_remaining"] = 7
    home_timeouts = 7
    away_timeouts = 7
    period = None
    df.reset_index(drop=True, inplace=True)

    for index,row in df.iterrows():
        if row["period"] != period:
            period = row["period"]
            if period >= 5:
                home_timeouts = home_timeouts + 2 if modern else home_timeouts + 1
                away_timeouts = away_timeouts + 2 if modern else away_timeouts + 1
        if is_team_timeout(row):
            if row["is_home_event"] == 1:
                home_timeouts -= 1
            elif row["is_home_event"] == 0:
                away_timeouts -= 1
        df.loc[index,"home_timeouts_remaining"] = home_timeouts
        df.loc[index,"away_timeouts_remaining"] = away_timeouts
        

def is_team_timeout(row:pd.Series) -> bool:
    return pd.notna(row["actionType"]) and row["subType"] in ["Regular","Short"]

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

