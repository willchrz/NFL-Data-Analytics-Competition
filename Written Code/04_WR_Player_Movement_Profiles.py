# NFL BIG DATA BOWL 2027
# WR PLAYER MOVEMENT PROFILES

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"W:\nfl-big-data-bowl-2027")

OUTPUT_DIR = PROJECT_DIR / "Output Data"
CODE_DIR = PROJECT_DIR / "Written Code"


movement_features = pd.read_csv(
    OUTPUT_DIR / "WR_Combine_Movement_Features.csv"
)

player_profiles = (
    movement_features.groupby(
        ["nfl_id", "draft_year", "display_name"],
        as_index=False
    )
    .agg(
        overall_average_speed=("average_speed", "mean"),
        overall_maximum_speed=("maximum_speed", "max"),
        overall_average_acceleration=(
            "average_acceleration", "mean"
        ),
        overall_maximum_acceleration=(
            "maximum_acceleration", "max"
        ),
        average_acceleration_variability=(
            "acceleration_std", "mean"
        ),
        average_distance_per_frame=(
            "average_distance_per_frame", "mean"
        ),
        total_distance_across_drills=(
            "total_distance", "sum"
        ),
        total_drill_attempts=("attempt", "count")
    )
)

player_profiles.to_csv(
    OUTPUT_DIR / "WR_Player_Movement_Profiles.csv",
    index=False
)

print("Number of players:", len(player_profiles))

print(player_profiles.head(10))

print("\nMovement profiles saved successfully.")