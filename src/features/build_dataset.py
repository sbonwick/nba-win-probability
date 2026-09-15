import argparse
import re
import pandas as pd
from pathlib import Path

from src.data.data_constants import GAME_TYPES, SEASONS
import src.data.cache as cache
import src.utils.io as io
from src.utils.logging_utils import get_logger

from src.features.game_state import prepare_tables
from src.data.fetch_pbp import process_game_ids,validate_game_ids_for_season


logger = get_logger(__name__)

def valid_season(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise argparse.ArgumentTypeError(
            "Season must be in format YYYY-YY, e.g. 2023-24"
        )
    return value

def main():
    parser = argparse.ArgumentParser(
        description="Processing data from games into intended format"
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
                    failure = process_interim(game_id, season, game_type)
                    if failure:
                        all_failures.append(failure)
    else:
        for season in seasons_to_fetch:
            for game_type in season_types_to_fetch:
                failures = []
                game_ids_to_fetch = process_game_ids(season, game_type)

                for game_id in game_ids_to_fetch:
                    failure = process_interim(game_id, season, game_type)
                    if failure:
                        failures.append(failure)
                        all_failures.append(failure)

                if failures:
                    failure_df = pd.DataFrame(failures)
                    failure_path = cache.interimDataFrameFailurePath(
                        season=season,
                        game_type=game_type,
                    )
                    io.write_df_csv(failure_df, failure_path)

    if all_failures:
        logger.error("Completed with %d failures", len(all_failures))

def process_interim(game_id: str, season: str, game_type: str) -> dict | None:
    try:
        path = cache.interimDataFramePath(season=season,game_id=game_id,game_type=game_type)
        if cache.isCached(path):
            return None
        df = prepare_tables(game_id=game_id,season_type=game_type,season=season)
        save_interim_dataframe(df=df,path=path)
        logger.info(
            "Processed data for for game_id=%s, season=%s, game_type=%s",
            game_id,
            season,
            game_type,
        )
        return None
    
    except Exception as e:
        logger.error(
            "Failed data processing for game_id=%s, season=%s, game_type=%s: %s",
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

def save_interim_dataframe(df:pd.DataFrame,path:Path):
    io.write_df_csv(df,path)

if __name__ == "__main__":
    main()