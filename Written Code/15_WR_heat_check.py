import pandas as pd
from pathlib import Path

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
SOURCE = ROOT / "Source Data"

player_play = pd.read_csv(
    SOURCE / "player_play.csv",
    nrows=5
)

games = pd.read_csv(
    SOURCE / "games.csv",
    nrows=5
)

print("\nPLAYER_PLAY COLUMNS")
for column in player_play.columns:
    print(column)

print("\nGAMES COLUMNS")
for column in games.columns:
    print(column)
