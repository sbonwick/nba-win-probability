import src.utils.io as io
import argparse
import re
from pathlib import Path

import pandas as pd

import src.data.cache as cache
from src.utils.logging_utils import get_logger
from nba_api.stats.endpoints import BoxScoreSummaryV3
from data.data_constants import GAME_TYPES, SEASONS

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
        description="Fetch box score data for one or more games"
    )
    parser.add_argument("--game-id", nargs="+", required=False, type=str)
    parser.add_argument("--season", nargs="+", required=False, type=valid_season)
    parser.add_argument(
        "--season-type",
        nargs="+",
        required=False,
        choices=GAME_TYPES,
    )
    args = parser.parse_args()

    if args.game_id:
        if not args.season or not args.season_type:
            raise ValueError(
                "If --game-id is provided, both --season and --season-type must also be specified."
            )

        if len(args.season) != 1 or len(args.season_type) != 1:
            raise ValueError(
                "When --game-id is provided, exactly one --season and one --season-type must be specified."
            )

        game_ids_to_fetch = args.game_id
        seasons_to_fetch = args.season
        season_types_to_fetch = args.season_type

        validate_game_ids_for_season(
            game_ids_to_fetch,
            seasons_to_fetch[0],
            season_types_to_fetch[0],
        )
    else:
        if not args.season or not args.season_type:
            raise ValueError(
                "If --game-id is not provided, both --season and --season-type must be specified."
            )
        seasons_to_fetch = args.season
        season_types_to_fetch = args.season_type

    all_failures = []

    if args.game_id:
        for season in seasons_to_fetch:
            for game_type in season_types_to_fetch:
                for game_id in game_ids_to_fetch:
                    failure = process_box_score(game_id, season, game_type)
                    if failure:
                        all_failures.append(failure)
    else:
        for season in seasons_to_fetch:
            for game_type in season_types_to_fetch:
                failures = []
                game_ids_to_fetch = process_game_ids(season, game_type)

                for game_id in game_ids_to_fetch:
                    failure = process_box_score(game_id, season, game_type)
                    if failure:
                        failures.append(failure)
                        all_failures.append(failure)

                if failures:
                    failure_df = pd.DataFrame(failures)
                    failure_path = cache.boxScoreFailurePath(
                        season=season,
                        game_type=game_type,
                    )
                    io.write_df_csv(failure_df, failure_path)

    if all_failures:
        logger.error("Completed with %d failures", len(all_failures))


def process_box_score(game_id: str, season: str, game_type: str) -> dict | None:
    try:
        path = cache.boxScorePath(
            season=season,
            game_id=game_id,
            game_type=game_type,
        )

        if cache.isCached(path):
            return None

        data = fetch_box_score_from_api(game_id)
        validate_box_score_dataframe(data, game_id)
        data = clean_box_score_dataframe(data)
        save_box_score_dataframe(data, path)

        logger.info(
            "Saved box score for game_id=%s, season=%s, game_type=%s",
            game_id,
            season,
            game_type,
        )
        return None

    except Exception as e:
        logger.error(
            "Failed box score fetch for game_id=%s, season=%s, game_type=%s: %s",
            game_id,
            season,
            game_type,
            str(e),
        )
        return {
            "game_id": game_id,
            "season": season,
            "season_type": game_type,
            "error_message": str(e),
        }


def fetch_box_score_from_api(game_id: str) -> pd.DataFrame:
    reader = BoxScoreSummaryV3(game_id=game_id)

    game_summary = reader.game_summary.get_data_frame()
    line_score = reader.line_score.get_data_frame()

    home_id = game_summary.loc[0, "homeTeamId"]

    line_score["IS_HOME"] = line_score["teamId"] == home_id
    line_score["WON"] = line_score["score"] == line_score["score"].max()

    line_score = line_score.rename(
        columns={
            "gameId": "GAME_ID",
            "teamId": "TEAM_ID",
            "teamTricode": "TEAM_ABBREVIATION",
            "score": "PTS",
            "teamWins": "WINS",
            "teamLosses": "LOSSES",
        }
    )

    return line_score


def validate_game_ids_for_season(
    game_ids: list[str],
    season: str,
    season_type: str,
) -> None:
    valid_game_ids = set(process_game_ids(season, season_type))
    invalid_game_ids = [game_id for game_id in game_ids if game_id not in valid_game_ids]

    if invalid_game_ids:
        raise ValueError(
            f"The following game IDs are not valid for season={season}, "
            f"season_type={season_type}: {invalid_game_ids}"
        )


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
    io.write_df_csv(df, output_path)


def process_game_ids(season: str, season_type: str) -> list[str]:
    path = cache.gameIDPath(season, season_type)
    if not cache.isCached(path):
        raise ValueError(
            f"Game ID file does not exist for season {season} and type {season_type}. Expected at {path}"
        )
    df = io.read_csv(path)
    return df["GAME_ID"].astype(str).tolist()


if __name__ == "__main__":
    main()