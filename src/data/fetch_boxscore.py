import src.utils.io as io
import argparse
import re
from pathlib import Path

import pandas as pd

import src.data.cache as cache
from src.utils.logging_utils import get_logger
from nba_api.stats.endpoints import BoxScoreSummaryV3

BOX_SCORE_COLUMNS = [
    "GAME_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "PTS",
    "WINS",
    "LOSSES",
    "IS_HOME",
    "WON",
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
        description="Fetch box score summary data for a specific game"
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

    logger.info("Starting box score fetch for game_id=%s", game_id)

    path = cache.boxScorePath(season=season, game_id=game_id, game_type=game_type)

    if cache.isCached(path):
        logger.info("Skipping fetch because cached file already exists at %s", path)
        return

    data = fetch_box_score_from_api(game_id)
    logger.info("Fetched %d raw rows from BoxScoreSummaryV2", len(data))

    data = clean_box_score_dataframe(data)

    validate_box_score_dataframe(data, game_id)
    save_box_score_dataframe(data, path)


def fetch_box_score_from_api(game_id: str) -> pd.DataFrame:
    reader = BoxScoreSummaryV3(game_id=game_id)

    game_summary = reader.game_summary.get_data_frame()   
    line_score = reader.line_score.get_data_frame()  

    home_id = game_summary.loc[0, "homeTeamId"]

    line_score["IS_HOME"] = line_score["teamId"] == home_id
    line_score["WON"] = line_score["score"] == line_score["score"].max()

    line_score = line_score.rename(columns={
        "gameId": "GAME_ID",
        "teamId": "TEAM_ID",
        "teamTricode": "TEAM_ABBREVIATION",
        "score": "PTS",
        "teamWins": "WINS",
        "teamLosses": "LOSSES",
    })

    return line_score


def clean_box_score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df[BOX_SCORE_COLUMNS]


def validate_box_score_dataframe(df: pd.DataFrame, game_id: str) -> None:
    if df.empty:
        raise ValueError("Empty DataFrame")

    missing_columns = [col for col in BOX_SCORE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in DataFrame: {missing_columns}"
        )

    bad_game_ids = df.loc[df["GAME_ID"].astype(str) != game_id, "GAME_ID"].unique()
    if len(bad_game_ids) > 0:
        raise ValueError(
            f"DataFrame contains unexpected GAME_ID values: {bad_game_ids.tolist()} "
            f"(expected only {game_id})"
        )


def save_box_score_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Saving cleaned box score DataFrame to %s", output_path)
    io.write_df_csv(df, output_path)


if __name__ == "__main__":
    main()