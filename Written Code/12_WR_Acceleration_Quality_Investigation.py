
import pandas as pd
import numpy as np
from pathlib import Path

# Project folders
ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

# Input file
tracking_file = OUTPUT / "WR_Outlier_Tracking_Sequences.csv"

# Output files
player_tracking_file = OUTPUT / "WR_Player_Acceleration_Investigation.csv"
acceleration_summary_file = OUTPUT / "WR_Acceleration_Quality_Summary.csv"

# Load tracking sequences
tracking = pd.read_csv(tracking_file)

print("\nWR ACCELERATION QUALITY INVESTIGATION")
print("Tracking observations:", len(tracking))
print("Columns:", tracking.columns.tolist())

# Check entity types
print("\nENTITY TYPE COUNTS")
print(tracking["entity_type"].value_counts(dropna=False))


# Keep PLAYER observations only
player_tracking = tracking[
    tracking["entity_type"] == "PLAYER"
].copy()

# Sort observations chronologically within each player/event
player_tracking = player_tracking.sort_values(
    ["nfl_id", "event_id", "time"]
)

# Convert time to datetime
player_tracking["time"] = pd.to_datetime(
    player_tracking["time"]
)

# Calculate time between consecutive frames
player_tracking["time_diff"] = (
    player_tracking.groupby(["nfl_id", "event_id"])["time"]
    .diff()
    .dt.total_seconds()
)

# Calculate speed change between consecutive frames
player_tracking["speed_diff"] = (
    player_tracking.groupby(["nfl_id", "event_id"])["s"]
    .diff()
)

# Calculate longitudinal acceleration
player_tracking["longitudinal_acceleration"] = (
    player_tracking["speed_diff"] /
    player_tracking["time_diff"]
)

# Save player-only tracking observations
player_tracking.to_csv(
    player_tracking_file,
    index=False
)

print("\nPLAYER TRACKING OBSERVATIONS")
print("Player observations:", len(player_tracking))
print("Saved:", player_tracking_file)


# Summarize acceleration by player and drill
acceleration_summary = (
    player_tracking.groupby(
        ["nfl_id", "display_name", "draft_year", "drill_name"]
    )
    .agg(
        observations=("s", "count"),
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
        ),
        maximum_speed=("s", "max")
    )
    .reset_index()
)

# Save summary
acceleration_summary.to_csv(
    acceleration_summary_file,
    index=False
)

print("\nACCELERATION SUMMARY")
print(acceleration_summary.head(10))

print("\nSAVED SUMMARY")
print(acceleration_summary_file)

print("\nSCRIPT 12 COMPLETED")
