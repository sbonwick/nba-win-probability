SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

GAME_TYPES = ["Regular Season", "Playoffs"]

TRAINING_SEASONS = [    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23"]

VALIDATION_SEASONS = ["2023-24","2024-25"]

TEST_SEASONS = ["2024-25"]

CSV_COLUMNS = ["gameId","period","scoreHome","scoreAway","home_wins","home_losses","home_won","away_wins","away_losses","playoffs","time_elapsed","time_remaining","scoreDifferential",
               "home_team_fouls_period","away_team_fouls_period","home_in_penalty","away_in_penalty","home_possession","home_timeouts_remaining","away_timeouts_remaining",
               "home_win_pct","away_win_pct","win_pct_diff","is_overtime"]

BASELINE_FEATURES = [
    "period",
    "scoreDifferential",
    "home_games_played",
    "away_games_played",
    "home_win_pct",
    "away_win_pct",
    "playoffs",
    "time_remaining",
    "home_team_fouls_period",
    "away_team_fouls_period",
    "home_in_penalty",
    "away_in_penalty",
    "home_possession",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "is_overtime",
]

INTEGER_COLUMNS = [
    "period",
    "scoreHome",
    "scoreAway",
    "home_wins",
    "home_losses",
    "home_won",
    "away_wins",
    "away_losses",
    "playoffs",
    "scoreDifferential",
    "home_team_fouls_period",
    "away_team_fouls_period",
    "home_in_penalty",
    "away_in_penalty",
    "home_possession",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "is_overtime",
]
FLOAT_COLUMNS = [
    "time_elapsed",
    "time_remaining",
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
]