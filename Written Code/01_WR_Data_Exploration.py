# NFL BIG DATA BOWL 2027
# WR DATA EXPLORATION

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(r"W:\nfl-big-data-bowl-2027")

SOURCE_DIR = PROJECT_DIR / "Source Data"

OUTPUT_DIR = PROJECT_DIR / "Output Data"

CODE_DIR = PROJECT_DIR / "Written Code"

players = pd.read_csv(SOURCE_DIR / "players.csv")

combine_results = pd.read_csv(SOURCE_DIR / "combine_results.csv")

combine_tracking = pd.read_csv(SOURCE_DIR / "combine_tracking.csv")

player_play = pd.read_csv(SOURCE_DIR / "player_play.csv")

career_successes = pd.read_csv(
    SOURCE_DIR / "player_career_successes.csv"
)

games = pd.read_csv(SOURCE_DIR / "games.csv")

game_tracking = pd.read_csv(
    SOURCE_DIR / "game_tracking_2023.csv"
)

datasets = {
    "players": players,
    "combine_results": combine_results,
    "combine_tracking": combine_tracking,
    "player_play": player_play,
    "career_successes": career_successes,
    "games": games,
    "game_tracking": game_tracking
}

for name, df in datasets.items():

    print("\n" + "=" * 60)

    print(f"DATASET: {name}")

    print("=" * 60)

    print("Rows:", df.shape[0])

    print("Columns:", df.shape[1])

    print("\nColumn names:")

    print(df.columns.tolist())

    print("\nFirst 5 rows:")

    print(df.head())

# Identify wide receivers
wr_players = players[
    players["nfl_position"] == "WR"
].copy()

print("\n" + "=" * 60)

print("WIDE RECEIVER SAMPLE")

print("=" * 60)

print("Number of WRs:", len(wr_players))

print("\nWRs by draft year:")

print(
    wr_players["draft_year"]
    .value_counts()
    .sort_index()
)

wr_players.to_csv(
    OUTPUT_DIR / "WR_Player_List.csv",
    index=False
)

print(
    wr_players[
        [
            "nfl_id",
            "display_name",
            "draft_year",
            "draft_overall_pick"
        ]
    ].head(15)
)

wr_players.to_csv(
    OUTPUT_DIR / "WR_Player_List.csv",
    index=False
)

print("\nWR player list saved successfully.")
