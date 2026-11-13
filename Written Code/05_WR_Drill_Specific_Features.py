# NFL BIG DATA BOWL 2027
# WR DRILL-SPECIFIC MOVEMENT FEATURES

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT_DIR = PROJECT_DIR / "Output Data"
movement_features = pd.read_csv(
    OUTPUT_DIR / "WR_Combine_Movement_Features.csv"
)
drill_features = (
    movement_features.groupby(
        ["nfl_id", "draft_year", "display_name", "drill_name"]
    )
    .agg(
        drill_average_speed=("average_speed", "mean"),
        drill_maximum_speed=("maximum_speed", "max"),
        drill_average_acceleration=(
            "average_acceleration", "mean"
        ),
        drill_maximum_acceleration=(
            "maximum_acceleration", "max"
        ),
        drill_acceleration_variability=(
            "acceleration_std", "mean"
        ),
        drill_total_distance=("total_distance", "sum"),
        drill_attempts=("attempt", "count")
    )
    .reset_index()
)
drill_features_wide = drill_features.pivot(
    index=["nfl_id", "draft_year", "display_name"],
    columns="drill_name",
    values=[
        "drill_average_speed",
        "drill_maximum_speed",
        "drill_average_acceleration",
        "drill_maximum_acceleration"
    ]
)
# Flatten column names
drill_features_wide.columns = [
    f"{feature}_{drill}"
    for feature, drill in drill_features_wide.columns
]
drill_features_wide = drill_features_wide.reset_index()
drill_features_wide.to_csv(
    OUTPUT_DIR / "WR_Drill_Specific_Features.csv",
    index=False
)

print("\nWR DRILL-SPECIFIC FEATURES")

print("Number of players:", len(drill_features_wide))
print("Number of columns:", len(drill_features_wide.columns))

print(drill_features_wide.head())

print("\nDrill-specific features saved successfully.")