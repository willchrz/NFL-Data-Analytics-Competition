# NFL BIG DATA BOWL 2027
# WR Combine Athletic Profile Integration

import pandas as pd
from pathlib import Path

project_folder = Path(r"W:\nfl-big-data-bowl-2027")

source_folder = project_folder / "Source Data"
output_folder = project_folder / "Output Data"

players = pd.read_csv(
    output_folder / "WR_Player_List.csv"
)

combine = pd.read_csv(
    source_folder / "combine_results.csv"
)

movement = pd.read_csv(
    output_folder / "WR_Player_Movement_Aggregated.csv"
)

print("WR players:", len(players))
print("Combine records:", len(combine))
print("Movement records:", len(movement))

combine_wr = combine.merge(
    players[["nfl_id", "draft_year"]],
    on=["nfl_id", "draft_year"],
    how="inner"
)

print("\nWR combine records:", len(combine_wr))

athletic_profile = players.merge(
    combine_wr,
    on=["nfl_id", "draft_year"],
    how="left",
    suffixes=("", "_combine")
)

athletic_profile = athletic_profile.merge(
    movement,
    on=["nfl_id", "display_name", "draft_year"],
    how="left",
    suffixes=("", "_movement")
)

output_file = (
    output_folder / "WR_Combine_Athletic_Profile.csv"
)

athletic_profile.to_csv(
    output_file,
    index=False
)

print("\nPreview:")
print(athletic_profile.head(10))