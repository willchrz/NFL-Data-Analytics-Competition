
# NFL BIG DATA BOWL 2027
# WR RAW TRACKING SEQUENCE EXPLORATION

import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(r"W:\nfl-big-data-bowl-2027")
SOURCE_DIR = PROJECT_DIR / "Source Data"
OUTPUT_DIR = PROJECT_DIR / "Output Data"

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

wr_players = pd.read_csv(
    OUTPUT_DIR / "WR_Player_List.csv"
)

tracking = pd.read_csv(
    SOURCE_DIR / "combine_tracking.csv"
)

# --------------------------------------------------
# 3. FILTER WIDE RECEIVERS
# --------------------------------------------------

wr_tracking = tracking[
    tracking["nfl_id"].isin(wr_players["nfl_id"])
].copy()

# --------------------------------------------------
# 4. INSPECT TRACKING STRUCTURE
# --------------------------------------------------

print("\nRAW TRACKING STRUCTURE")

print("\nColumns:")
print(wr_tracking.columns.tolist())

print("\nNumber of observations:", len(wr_tracking))

print("\nNumber of players:", wr_tracking["nfl_id"].nunique())

print("\nUnique event IDs:", wr_tracking["event_id"].nunique())

print("\nUnique drill types:")
print(wr_tracking["drill_type"].value_counts())

print("\nUnique drill names:")
print(wr_tracking["drill_name"].value_counts())

# --------------------------------------------------
# 5. CHECK ACCELERATION VALUES
# --------------------------------------------------

print("\nACCELERATION SUMMARY")

print(wr_tracking["a"].describe())

print("\nAcceleration below zero:")
print((wr_tracking["a"] < 0).sum())

print("\nAcceleration equal to zero:")
print((wr_tracking["a"] == 0).sum())

# --------------------------------------------------
# 6. CHECK TRACKING TIME INTERVALS
# --------------------------------------------------

print("\nTIME INTERVAL EXAMPLE")

sample_event = wr_tracking["event_id"].iloc[0]

sample_tracking = wr_tracking[
    wr_tracking["event_id"] == sample_event
].copy()

# Convert timestamps to datetime
sample_tracking["time_datetime"] = pd.to_datetime(
    sample_tracking["time"]
)

# Sort chronologically
sample_tracking = sample_tracking.sort_values(
    "time_datetime"
)

# Calculate elapsed time in seconds
sample_tracking["time_difference"] = (
    sample_tracking["time_datetime"].diff().dt.total_seconds()
)

print(sample_tracking[
    ["event_id", "time", "s", "a", "dis",
     "dir", "x", "y", "time_difference"]
].head(20))

# --------------------------------------------------
# 7. CHECK EVENT STRUCTURE
# --------------------------------------------------

wr_tracking["time_datetime"] = pd.to_datetime(
    wr_tracking["time"]
)

event_summary = (
    wr_tracking.groupby(
        ["nfl_id", "event_id", "drill_name", "attempt"]
    )
    .agg(
        observations=("time", "count"),
        start_time=("time_datetime", "min"),
        end_time=("time_datetime", "max"),
        maximum_speed=("s", "max"),
        maximum_acceleration=("a", "max")
    )
    .reset_index()
)

print("\nEVENT STRUCTURE")

print("\nEvent summary:")
print(event_summary.head(10))

print("\nObservations per event:")
print(event_summary["observations"].describe())

# --------------------------------------------------
# 8. SAVE EVENT SUMMARY
# --------------------------------------------------

event_summary.to_csv(
    OUTPUT_DIR / "WR_Combine_Event_Summary.csv",
    index=False
)

print("\nEvent summary saved successfully.")
