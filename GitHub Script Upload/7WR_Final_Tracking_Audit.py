# NFL BIG DATA BOWL 2027
# OMG ITS THE FINAL WR FEATURE LIST (he doesn't know)

import pandas as pd
import numpy as np
from pathlib import Path

# Project folders
ROOT = Path(r"W:\nfl-big-data-bowl-2027")
SOURCE = ROOT / "Source Data"
OUTPUT = ROOT / "Output Data"

# files
tracking_file = SOURCE / "combine_tracking.csv"
players_file = OUTPUT / "WR_Player_List.csv"
output_file = OUTPUT / "WR_Final_Movement_Features.csv"

players = pd.read_csv(players_file)
tracking = pd.read_csv(tracking_file)

print("WR players:", players["nfl_id"].nunique())
print("Tracking observations:", len(tracking))

# Keep observations belonging to WRs
tracking_wr = tracking[
    tracking["nfl_id"].isin(players["nfl_id"])
].copy()

# Keep PLAYER entities only
tracking_wr = tracking_wr[
    tracking_wr["entity_type"] == "PLAYER"
].copy()

# Add player names
tracking_wr = tracking_wr.merge(
    players[["nfl_id", "display_name"]],
    on="nfl_id",
    how="left"
)

print("Observations:", len(tracking_wr))
print("Unique players:", tracking_wr["nfl_id"].nunique())
print("Entity types:", tracking_wr["entity_type"].unique())

# Convert tracking time to datetime
tracking_wr["time"] = pd.to_datetime(tracking_wr["time"])

# Sort chronologically within each player and event
tracking_wr = tracking_wr.sort_values(
    ["nfl_id", "event_id", "time"]
)

# Calculate time difference between frames
tracking_wr["time_diff"] = (
    tracking_wr.groupby(["nfl_id", "event_id"])["time"]
    .diff()
    .dt.total_seconds()
)

# Calculate speed difference
tracking_wr["speed_diff"] = (
    tracking_wr.groupby(["nfl_id", "event_id"])["s"]
    .diff()
)

# Calculate longitudinal acceleration
tracking_wr["longitudinal_acceleration"] = (
    tracking_wr["speed_diff"] /
    tracking_wr["time_diff"]
)

print(
    tracking_wr["longitudinal_acceleration"]
    .describe()
)

# Aggregate movement features by player
movement_features = (
    tracking_wr.groupby(["nfl_id", "display_name"])
    .agg(
        tracking_observations=("s", "count"),
        average_speed=("s", "mean"),
        maximum_speed=("s", "max"),
        speed_std=("s", "std"),
        average_raw_acceleration=("a", "mean"),
        maximum_raw_acceleration=("a", "max"),
        average_longitudinal_acceleration=(
            "longitudinal_acceleration", "mean"
        ),
        maximum_longitudinal_acceleration=(
            "longitudinal_acceleration", "max"
        ),
        minimum_longitudinal_acceleration=(
            "longitudinal_acceleration", "min"
        )
    )
    .reset_index()
)

# Save final movement features
movement_features.to_csv(output_file, index=False)

print(movement_features.head())

print("\nPlayers:", movement_features["nfl_id"].nunique())
print("Columns:", len(movement_features.columns))
print("Saved:", output_file)

