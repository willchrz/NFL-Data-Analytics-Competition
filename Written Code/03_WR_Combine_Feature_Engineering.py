# NFL BIG DATA BOWL 2027
# WR COMBINE FEATURE ENGINEERING

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(r"W:\nfl-big-data-bowl-2027")

SOURCE_DIR = PROJECT_DIR / "Source Data"

OUTPUT_DIR = PROJECT_DIR / "Output Data"

wr_players = pd.read_csv(
    OUTPUT_DIR / "WR_Player_List.csv"
)

combine_tracking = pd.read_csv(
    SOURCE_DIR / "combine_tracking.csv"
)

wr_tracking = combine_tracking[
    combine_tracking["nfl_id"].isin(wr_players["nfl_id"])
].copy()

group_columns = [
    "nfl_id",
    "draft_year",
    "drill_name",
    "attempt"
]

movement_features = (
    wr_tracking
    .groupby(group_columns)
    .agg(
        average_speed=("s", "mean"),
        maximum_speed=("s", "max"),
        average_acceleration=("a", "mean"),
        maximum_acceleration=("a", "max"),
        acceleration_std=("a", "std"),
        average_distance_per_frame=("dis", "mean"),
        total_distance=("dis", "sum"),
        observations=("a", "count")
    )
    .reset_index()
)

movement_features = movement_features.merge(
    wr_players[
        ["nfl_id", "display_name", "draft_overall_pick"]
    ],
    on="nfl_id",
    how="left"
)

movement_features.to_csv(
    OUTPUT_DIR / "WR_Combine_Movement_Features.csv",
    index=False
)

print("Rows:", len(movement_features))

print(
    movement_features[
        [
            "display_name",
            "drill_name",
            "attempt",
            "average_speed",
            "maximum_speed",
            "average_acceleration",
            "maximum_acceleration"
        ]
    ].head(20)
)

print("\nFeatures saved successfully.")
