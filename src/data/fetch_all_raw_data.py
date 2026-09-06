import subprocess
import argparse
import re


def valid_season(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise argparse.ArgumentTypeError(
            "Season must be in format YYYY-YY, e.g. 2023-24"
        )
    return value


def run_command(command: list[str]) -> None:
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch all raw NBA data")
    parser.add_argument("--season", nargs="+", required=False, type=valid_season)
    parser.add_argument("--season-type", nargs="+", required=False)
    args = parser.parse_args()

    seasons = args.season if args.season else []    
    season_types = args.season_type if args.season_type else []

    game_id_command = ["python", "-m", "src.data.fetch_game_ids"]
    pbp_command = ["python", "-m", "src.data.fetch_pbp"]
    boxscore_command = ["python", "-m", "src.data.fetch_boxscore"]

    if seasons:
        game_id_command += ["--season", *seasons]
        pbp_command += ["--season", *seasons]
        boxscore_command += ["--season", *seasons]

    if season_types:
        game_id_command += ["--season-type", *season_types]
        pbp_command += ["--season-type", *season_types]
        boxscore_command += ["--season-type", *season_types]

    run_command(game_id_command)
    run_command(pbp_command)
    run_command(boxscore_command)


if __name__ == "__main__":
    main()