# NFL BIG DATA BOWL 2027
# WR Drill Specific Features might be good idk

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

EVENT_FILE = OUTPUT / "WR_Advanced_Event_Features.csv"
SUMMARY_FILE = OUTPUT / "WR_Combine_Event_Summary.csv"
OUTPUT_FILE = OUTPUT / "WR_Drill_Specific_Movement_Features.csv"

events = pd.read_csv(EVENT_FILE)
summary = pd.read_csv(SUMMARY_FILE)


events = events.merge(
    summary[["nfl_id", "event_id", "drill_name", "attempt"]],
    on=["nfl_id", "event_id"],
    how="left"
)

print("\nDrill coverage:")
print(events["drill_name"].value_counts().head(20))

metrics = [
    "maximum_speed",
    "average_speed",
    "maximum_acceleration",
    "average_acceleration",
    "maximum_longitudinal_acceleration",
    "time_to_peak_speed",
    "first_second_average_speed",
    "first_second_maximum_speed",
    "first_second_average_acceleration",
    "duration"
]

metrics = [m for m in metrics if m in events.columns]

drill_features = events.groupby(
    ["nfl_id", "display_name", "draft_year", "drill_name"]
)[metrics].agg(["mean", "max", "std"])


drill_features.columns = [
    f"{col[0]}_{col[1]}"
    for col in drill_features.columns
]

drill_features = drill_features.reset_index()

wide_features = drill_features.pivot(
    index=["nfl_id", "display_name", "draft_year"],
    columns="drill_name"
)

wide_features.columns = [
    f"{drill}_{feature}"
    for feature, drill in wide_features.columns
]

wide_features = wide_features.reset_index()

attempt_features = events.groupby(
    ["nfl_id", "display_name", "draft_year", "drill_name"]
).agg(
    attempt_speed_std=("maximum_speed", "std"),
    attempt_acceleration_std=("maximum_acceleration", "std"),
    attempt_speed_mean=("maximum_speed", "mean"),
    attempt_acceleration_mean=("maximum_acceleration", "mean"),
    number_of_attempts=("event_id", "nunique")
).reset_index()

attempt_features["speed_consistency"] = (
    attempt_features["attempt_speed_std"] /
    attempt_features["attempt_speed_mean"]
)

attempt_features["acceleration_consistency"] = (
    attempt_features["attempt_acceleration_std"] /
    attempt_features["attempt_acceleration_mean"]
)

attempt_wide = attempt_features.pivot(
    index=["nfl_id", "display_name", "draft_year"],
    columns="drill_name",
    values=[
        "speed_consistency",
        "acceleration_consistency",
        "number_of_attempts"
    ]
)

attempt_wide.columns = [
    f"{drill}_{metric}"
    for metric, drill in attempt_wide.columns
]

attempt_wide = attempt_wide.reset_index()

final_features = wide_features.merge(
    attempt_wide,
    on=["nfl_id", "display_name", "draft_year"],
    how="left"
)

final_features.to_csv(OUTPUT_FILE, index=False)

print("\nOUTPUT SHAPE:", final_features.shape)

print("\nSAMPLE FEATURES:")
print(final_features.head())
print("\nSAVED:", OUTPUT_FILE)