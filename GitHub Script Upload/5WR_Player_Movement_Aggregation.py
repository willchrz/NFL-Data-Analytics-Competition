# NFL BIG DATA BOWL 2027
# WR Player-Level Movement Aggregation

import pandas as pd
from pathlib import Path

project_folder = Path(r"W:\nfl-big-data-bowl-2027")

source_folder = project_folder / "Source Data"
output_folder = project_folder / "Output Data"

event_features = pd.read_csv(
    output_folder / "WR_Advanced_Event_Features.csv"
)

player_movement = (
    event_features
    .groupby(
        ["nfl_id", "display_name", "draft_year"],
        as_index=False
    )
    .agg(
        # Number of drill events
        total_events=("event_id", "count"),

        # Overall movement
        average_event_speed=("average_speed", "mean"),
        maximum_event_speed=("maximum_speed", "max"),
        average_event_acceleration=("average_acceleration", "mean"),
        maximum_event_acceleration=("maximum_acceleration", "max"),

        # Longitudinal acceleration
        average_longitudinal_acceleration=(
            "maximum_longitudinal_acceleration", "mean"
        ),
        maximum_longitudinal_acceleration=(
            "maximum_longitudinal_acceleration", "max"
        ),
        minimum_longitudinal_acceleration=(
            "minimum_longitudinal_acceleration", "min"
        ),

        # Time to peak speed
        average_time_to_peak_speed=("time_to_peak_speed", "mean"),

        # First-second movement
        average_first_second_speed=(
            "first_second_average_speed", "mean"
        ),
        maximum_first_second_speed=(
            "first_second_maximum_speed", "max"
        ),
        average_first_second_acceleration=(
            "first_second_average_acceleration", "mean"
        ),

        # Drill duration
        average_event_duration=("duration", "mean"),
        maximum_event_duration=("duration", "max"),

        # Consistency across events
        speed_variability=("average_speed", "std"),
        acceleration_variability=("average_acceleration", "std")
    )
)

# Coefficient of variation for speed
player_movement["speed_coefficient_variation"] = (
    player_movement["speed_variability"]
    / player_movement["average_event_speed"]
)

# Coefficient of variation for acceleration
player_movement["acceleration_coefficient_variation"] = (
    player_movement["acceleration_variability"]
    / player_movement["average_event_acceleration"]
)

output_file = output_folder / "WR_Player_Movement_Aggregated.csv"

player_movement.to_csv(output_file, index=False)

print("Number of players:", len(player_movement))
print("Number of features:", len(player_movement.columns))

print("\nPreview:")
print(player_movement.head(10))

print("\nFeature summary:")
print(
    player_movement[
        [
            "average_event_speed",
            "maximum_event_speed",
            "average_first_second_speed",
            "maximum_first_second_speed",
            "average_first_second_acceleration"
        ]
    ].describe()
)

print("\nSaved to:", output_file)