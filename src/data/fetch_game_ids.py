import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder
from src.utils.logging_utils import get_logger
import src.utils.io as io
from pathlib import Path
import src.data.cache as cache
import argparse
import re

REQUIRED_GAME_ID_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "SEASON_ID",
]


def valid_season(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise argparse.ArgumentTypeError(
            "Season must be in format YYYY-YY, e.g. 2023-24"
        )
    return value

def main():
    parser = argparse.ArgumentParser(description="Fetch game IDs for a season with season type")
    parser.add_argument("--season",required=True)
    parser.add_argument("--season-type",required=True,type=valid_season, choices=["Regular Season","Playoffs"])
    args = parser.parse_args()
    season = args.season
    season_type = args.season_type
    path = cache.gameIDPath(season,season_type)
    if cache.isCached(path):
        print(f"File at {path} already exists")
        return
    data = fetch_game_ids_from_api(season,season_type)
    data = clean_game_ids_dataframe(data,season,season_type)
    validate_game_ids_dataframe(data)
    save_game_ids_dataframe(data,path)
    
    print(
        f"Saved {len(data)} game IDs for season {season} "
        f"and season type '{season_type}' to {path}"
    )

    return

def fetch_game_ids_from_api(season: str, season_type: str) -> pd.DataFrame:
    finder = LeagueGameFinder(player_or_team_abbreviation="T",season_nullable=season,season_type_nullable=season_type)
    df = finder.league_game_finder_results.get_data_frame()
    return df 

def clean_game_ids_dataframe(df: pd.DataFrame, season:str,season_type:str):
    df = df[df["SEASON_ID"] == season]
    df = df[df["SEASON_TYPE"] == season_type]
    df = df.drop_duplicates(subset="GAME_ID")
    df = df[REQUIRED_GAME_ID_COLUMNS]
    return df

def validate_game_ids_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Empty DataFrame")

    missing_columns = []

    for column in REQUIRED_GAME_ID_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in DataFrame: {missing_columns}"
        )

def save_game_ids_dataframe(df:pd.DataFrame, output_path: Path) -> None:
    io.write_df_csv(df,output_path)

if __name__ == "__main__":
    main()