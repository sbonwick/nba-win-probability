from src.data.data_constants import TRAINING_SEASONS, VALIDATION_SEASONS, TEST_SEASONS, GAME_TYPES, CSV_COLUMNS,INTEGER_COLUMNS,FLOAT_COLUMNS
from src.config import PROCESSED_DIR, INTERIM_DIR
from src.utils.logging_utils import get_logger

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

logger = get_logger(__name__)

PARQUET_SCHEMA = pa.schema(
    [pa.field("gameId",pa.string())] 
    + [pa.field(col,pa.int64()) for col in INTEGER_COLUMNS]
    + [pa.field(col,pa.float64()) for col in FLOAT_COLUMNS]
)

def get_game_files(seasons: list[str]) -> list[Path]:
    game_files :list[Path] = []
    for season in seasons:
        season_dir = INTERIM_DIR/season
        if not season_dir.exists():
            raise FileNotFoundError(f"Directory for interim {season} data not found")
        files = sorted(season_dir.rglob("*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No game CSVs found under {season_dir}")
        
        game_files.extend(files)
    return game_files

def prepare_game_frame(path:Path) -> pd.DataFrame:
    frame = pd.read_csv(path,dtype = {"gameId":"string"})

    missing = set(CSV_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(CSV_COLUMNS)

    if missing or extra:
        raise ValueError(
            f"Unexpected schema in {path}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    if frame["gameId"].isna().any() or frame["gameId"].eq("").any():
        raise ValueError(f"Missing gameId values in {path}")
    if frame["gameId"].nunique() != 1:
        raise ValueError(f"Expected exactly one gameId in {path}")

    for column in INTEGER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")

    for column in FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    return frame

def write_divided_parquet(data_type:str,seasons: list[str]) -> None:
    output_path = PROCESSED_DIR / f"{data_type}.parquet"
    game_files = get_game_files(seasons)
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    row_count = 0
    game_ids = set()

    with pq.ParquetWriter(output_path,PARQUET_SCHEMA,compression="zstd") as writer:
        for index, csv_path in enumerate(game_files,start=1):
            frame = prepare_game_frame(csv_path)
            game_id = str(frame["gameId"].iloc[0])

            if game_id in game_ids:
                raise ValueError(f"Duplicate gameId within {data_type}: {game_id}")

            game_ids.add(game_id)
            table = pa.Table.from_pandas(
                frame,
                schema = PARQUET_SCHEMA,
                preserve_index = False,
                safe = True
            )
            writer.write_table(table)
            row_count += len(frame)

            if index % 100 == 0 or index == len(game_files):
                logger.info(
                    "%s: wrote %d/%d games",
                    data_type,
                    index,
                    len(game_files),
                )

    logger.info(
        "Finished %s: %d games, %d game-state rows",
        output_path.name,
        len(game_ids),
        row_count,
        )


def main():
    write_divided_parquet("training",TRAINING_SEASONS)
    write_divided_parquet("validation",VALIDATION_SEASONS)
    write_divided_parquet("testing",TEST_SEASONS)


if __name__ == "__main__":
    main()

