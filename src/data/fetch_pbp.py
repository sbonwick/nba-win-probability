import src.utils.io as io
import argparse
import re
from pathlib import Path

import pandas as pd

import src.data.cache as cache
from src.utils.logging_utils import get_logger
from nba_api.stats.endpoints import PlayByPlayV3

PBP_COLUMNS = [
    "gameId",
    "actionNumber",
    "clock",
    "period",
    "teamId",
    "personId",
    "actionType",
    "subType",
    "description",
    "location",
    "scoreHome",
    "scoreAway",
]

logger = get_logger(__name__)

def valid_season(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise argparse.ArgumentTypeError(
            "Season must be in format YYYY-YY, e.g. 2023-24"
        )
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Fetch play-by-play data for a specific game"
    )
    parser.add_argument("--game-id", required=True, type=str)
    parser.add_argument("--season", required=True, type=valid_season)
    parser.add_argument(
        "--season-type",
        required=True,
        choices=["Regular Season", "Playoffs"],
    )
    args = parser.parse_args()

    game_id = args.game_id
    season = args.season
    game_type = args.season_type


    logger.info("Starting PBP fetch for game_id=%s", game_id)

    path = cache.pbpPath(season=season, game_id=game_id, game_type=game_type)

    if cache.isCached(path):
        logger.info("Skipping fetch because cached file already exists at %s", path)
        return

    data = fetch_pbp_from_api(game_id)
    logger.info("Fetched %d raw rows from PlayByPlayV3", len(data))

    validate_pbp_dataframe(data, game_id)
    data = clean_pbp_dataframe(data)

    save_pbp_dataframe(data, path)

def fetch_pbp_from_api(game_id:str) -> pd.DataFrame:
    reader = PlayByPlayV3(game_id=game_id,start_period=0,end_period=10)
    data = reader.get_data_frames()[0]
    return data

def clean_pbp_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    return df[PBP_COLUMNS]

def validate_pbp_dataframe(df: pd.DataFrame, game_id: str) -> None:
    if df.empty:
        raise ValueError("Empty DataFrame")

    missing_columns = [col for col in PBP_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in DataFrame: {missing_columns}"
        )

    bad_game_ids = df.loc[df["gameId"] != game_id, "gameId"].unique()
    if len(bad_game_ids) > 0:
        raise ValueError(
            f"DataFrame contains unexpected gameId values: {bad_game_ids.tolist()} "
            f"(expected only {game_id})"
        )

def save_pbp_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Saving cleaned PBP DataFrame to %s", output_path)
    io.write_df_csv(df, output_path)


if __name__ == "__main__":
    main()