import argparse
import re
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder

import src.data.cache as cache
import src.utils.io as io
from src.utils.logging_utils import get_logger

REQUIRED_GAME_ID_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "SEASON_ID",
]

logger = get_logger(__name__)


def valid_season(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise argparse.ArgumentTypeError(
            "Season must be in format YYYY-YY, e.g. 2023-24"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch game IDs for a season with season type"
    )
    parser.add_argument("--season", required=True, type=valid_season)
    parser.add_argument(
        "--season-type",
        required=True,
        choices=["Regular Season", "Playoffs"],
    )
    args = parser.parse_args()

    season = args.season
    season_type = args.season_type

    logger.info("Starting game ID fetch for season=%s, season_type=%s", season, season_type)

    path = cache.gameIDPath(season, season_type)

    if cache.isCached(path):
        logger.info("Skipping fetch because cached file already exists at %s", path)
        return

    data = fetch_game_ids_from_api(season, season_type)
    logger.info("Fetched %d raw rows from LeagueGameFinder", len(data))

    data = clean_game_ids_dataframe(data, season, season_type)
    logger.info("Cleaned data down to %d unique games", len(data))

    validate_game_ids_dataframe(data)
    save_game_ids_dataframe(data, path)

    logger.info(
        "Saved %d game IDs for season=%s, season_type=%s to %s",
        len(data),
        season,
        season_type,
        path,
    )


def fetch_game_ids_from_api(season: str, season_type: str) -> pd.DataFrame:
    logger.info("Requesting LeagueGameFinder data from nba_api")
    finder = LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season,
        season_type_nullable=season_type,
    )
    df = finder.league_game_finder_results.get_data_frame()
    return df


def clean_game_ids_dataframe(
    df: pd.DataFrame, season: str, season_type: str
) -> pd.DataFrame:
    logger.info("Cleaning game IDs DataFrame")
    df = df.drop_duplicates(subset="GAME_ID")
    df = df[REQUIRED_GAME_ID_COLUMNS]
    return df


def validate_game_ids_dataframe(df: pd.DataFrame) -> None:
    logger.info("Validating cleaned game IDs DataFrame")

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


def save_game_ids_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Saving cleaned game IDs DataFrame to %s", output_path)
    io.write_df_csv(df, output_path)


if __name__ == "__main__":
    main()
