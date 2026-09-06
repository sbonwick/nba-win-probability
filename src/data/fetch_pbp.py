import src.utils.io as io
import argparse
import re
from pathlib import Path

import pandas as pd

import src.data.cache as cache
from src.utils.logging_utils import get_logger
from nba_api.stats.endpoints import PlayByPlayV3
from src.data.data_constants import GAME_TYPES, SEASONS

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
        description="Fetch play-by-play data for one or more games"
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
        seasons_to_fetch = args.season if args.season else SEASONS
        season_types_to_fetch = args.season_type if args.season_type else GAME_TYPES

    all_failures = []

    if args.game_id:
        for season in seasons_to_fetch:
            for game_type in season_types_to_fetch:
                for game_id in game_ids_to_fetch:
                    failure = process_pbp(game_id, season, game_type)
                    if failure:
                        all_failures.append(failure)
    else:
        for season in seasons_to_fetch:
            for game_type in season_types_to_fetch:
                failures = []
                game_ids_to_fetch = process_game_ids(season, game_type)

                for game_id in game_ids_to_fetch:
                    failure = process_pbp(game_id, season, game_type)
                    if failure:
                        failures.append(failure)
                        all_failures.append(failure)

                if failures:
                    failure_df = pd.DataFrame(failures)
                    failure_path = cache.pbpFailurePath(
                        season=season,
                        game_type=game_type,
                    )
                    io.write_df_csv(failure_df, failure_path)

    if all_failures:
        logger.error("Completed with %d failures", len(all_failures))

def process_pbp(game_id: str, season: str, game_type: str) -> dict | None:
    try:
        path = cache.pbpPath(
            season=season,
            game_id=game_id,
            game_type=game_type,
        )

        if cache.isCached(path):
            return None

        data = fetch_pbp_from_api(game_id)
        validate_pbp_dataframe(data, game_id)
        data = clean_pbp_dataframe(data)
        save_pbp_dataframe(data, path)

        logger.info(
            "Saved PBP for game_id=%s, season=%s, game_type=%s",
            game_id,
            season,
            game_type,
        )
        return None

    except Exception as e:
        logger.error(
            "Failed PBP fetch for game_id=%s, season=%s, game_type=%s: %s",
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


def fetch_pbp_from_api(game_id: str) -> pd.DataFrame:
    reader = PlayByPlayV3(game_id=game_id, start_period=1, end_period=10)
    data = reader.get_data_frames()[0]
    return data


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


def clean_pbp_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
    io.write_df_csv(df, output_path)


def process_game_ids(season: str, season_type: str) -> list[str]:
    path = cache.gameIDPath(season, season_type)
    if not cache.isCached(path):
        raise ValueError(
            f"Game ID file does not exist for season {season} and type {season_type}. Expected at {path}"
        )
    df = io.read_csv(path, dtype={"GAME_ID": str})
    return df["GAME_ID"].tolist()


if __name__ == "__main__":
    main()