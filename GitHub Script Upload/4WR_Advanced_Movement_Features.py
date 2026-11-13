# NFL BIG DATA BOWL 2027
# WR ADVANCED MOVEMENT FEATURE ENGINEERING

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(r"W:\nfl-big-data-bowl-2027")
SOURCE_DIR = PROJECT_DIR / "Source Data"
OUTPUT_DIR = PROJECT_DIR / "Output Data"

wr_players = pd.read_csv(
    OUTPUT_DIR / "WR_Player_List.csv"
)

tracking = pd.read_csv(
    SOURCE_DIR / "combine_tracking.csv"
)

wr_tracking = tracking[
    tracking["nfl_id"].isin(wr_players["nfl_id"])
].copy()

# Convert timestamps
wr_tracking["time"] = pd.to_datetime(
    wr_tracking["time"]
)

wr_tracking = wr_tracking.sort_values(
    ["nfl_id", "event_id", "time"]
)

group_columns = ["nfl_id", "event_id"]

wr_tracking["elapsed_time"] = (
    wr_tracking["time"]
    - wr_tracking.groupby(group_columns)["time"].transform("min")
)

wr_tracking["elapsed_seconds"] = (
    wr_tracking["elapsed_time"].dt.total_seconds()
)

wr_tracking["time_difference"] = (
    wr_tracking.groupby(group_columns)["elapsed_seconds"].diff()
)

wr_tracking["speed_change"] = (
    wr_tracking.groupby(group_columns)["s"].diff()
)

wr_tracking["longitudinal_acceleration"] = (
    wr_tracking["speed_change"]
    / wr_tracking["time_difference"]
)

first_second = wr_tracking[
    wr_tracking["elapsed_seconds"] <= 1.0
].copy()

first_second_features = (
    first_second.groupby(group_columns)
    .agg(
        first_second_average_speed=("s", "mean"),
        first_second_maximum_speed=("s", "max"),
        first_second_average_acceleration=("a", "mean")
    )
    .reset_index()
)

event_features = (
    wr_tracking.groupby(group_columns)
    .agg(
        start_time=("time", "min"),
        duration=("elapsed_seconds", "max"),
        maximum_speed=("s", "max"),
        average_speed=("s", "mean"),
        maximum_acceleration=("a", "max"),
        average_acceleration=("a", "mean"),
        maximum_longitudinal_acceleration=(
            "longitudinal_acceleration", "max"
        ),
        minimum_longitudinal_acceleration=(
            "longitudinal_acceleration", "min"
        )
    )
    .reset_index()
)

peak_speed_times = (
    wr_tracking.loc[
        wr_tracking.groupby(group_columns)["s"].idxmax(),
        group_columns + ["elapsed_seconds"]
    ]
    .rename(
        columns={
            "elapsed_seconds": "time_to_peak_speed"
        }
    )
)

event_features = event_features.merge(
    peak_speed_times,
    on=group_columns,
    how="left"
)


event_features = event_features.merge(
    first_second_features,
    on=group_columns,
    how="left"
)

event_features = event_features.merge(
    wr_players[
        ["nfl_id", "display_name", "draft_year"]
    ],
    on="nfl_id",
    how="left"
)


event_features.to_csv(
    OUTPUT_DIR / "WR_Advanced_Event_Features.csv",
    index=False
)

print("Number of events:", len(event_features))

print(event_features.head(10))