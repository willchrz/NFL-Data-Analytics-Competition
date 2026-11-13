# NFL BIG DATA BOWL 2027
# Dataset that just has what we're looking for

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

LABEL_FILE = OUTPUT / "WR_Model_Relative.csv"
MOVEMENT_FILE = OUTPUT / "WR_Final_Movement_Features.csv"
DRILL_FILE = OUTPUT / "WR_Drill_Specific_Movement_Features.csv"

OUT_BASELINE = OUTPUT / "WR_Model_MovementOnly.csv"
OUT_DRILLS = OUTPUT / "WR_Model_MovementPlusDrills.csv"


labels = pd.read_csv(LABEL_FILE)
movement = pd.read_csv(MOVEMENT_FILE)
drills = pd.read_csv(DRILL_FILE)

print("\nInput shapes:")
print("Labels:", labels.shape)
print("Movement:", movement.shape)
print("Drills:", drills.shape)

print("\nLabel dataset columns:")
print(labels.columns.tolist())

TARGET = "target"

print("\nTarget variable:", TARGET)
print(labels[TARGET].value_counts(dropna=False))


# Remove overlapping movement variables already in labels
overlapping = [
    col for col in movement.columns
    if col in labels.columns and col != "nfl_id"
]

movement = movement.drop(columns=overlapping)

baseline = labels.merge(
    movement,
    on="nfl_id",
    how="left"
)

full_movement = baseline.merge(
    drills,
    on="nfl_id",
    how="left",
    suffixes=("", "_drill")
)

print("\nRemoved overlapping movement columns:", overlapping)
print("Baseline merge shape:", baseline.shape)
print("Full movement merge shape:", full_movement.shape)

for df in [baseline, full_movement]:

    df["acceleration_to_speed"] = (
        df["average_raw_acceleration"] /
        df["average_speed"].replace(0, np.nan)
    )

    df["peak_to_average_speed_ratio"] = (
        df["maximum_speed"] /
        df["average_speed"].replace(0, np.nan)
    )

    df["speed_variability_ratio"] = (
        df["speed_std"] /
        df["average_speed"].replace(0, np.nan)
    )

    df["longitudinal_acceleration_range"] = (
        df["maximum_longitudinal_acceleration"] -
        df["minimum_longitudinal_acceleration"]
    )

baseline = baseline[
    baseline["draft_year"].isin([2023, 2024])
].copy()

full_movement = full_movement[
    full_movement["draft_year"].isin([2023, 2024])
].copy()

# Remove unlabeled observations
baseline = baseline[baseline[TARGET].notna()].copy()
full_movement = full_movement[full_movement[TARGET].notna()].copy()

def movement_predictors(df):

    excluded = [
        TARGET,
        "nfl_id",
        "display_name",
        "draft_year"
    ]

    excluded_keywords = [
        "ngs",
        "college",
        "draft",
        "career",
        "receiving",
        "receptions",
        "yards",
        "starts",
        "snaps",
        "success",
        "label",
        "hit",
        "bust",
        "score"
    ]

    predictors = []

    for col in df.columns:

        if col in excluded:
            continue

        if any(keyword in col.lower()
               for keyword in excluded_keywords):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            predictors.append(col)

    return predictors

baseline_predictors = movement_predictors(baseline)
drill_predictors = movement_predictors(full_movement)

baseline_predictors = [
    c for c in baseline_predictors
    if baseline[c].isna().mean() <= 0.80
]

drill_predictors = [
    c for c in drill_predictors
    if full_movement[c].isna().mean() <= 0.80
]

baseline_model = baseline[
    ["nfl_id", "display_name", "draft_year", TARGET]
    + baseline_predictors
].copy()

drill_model = full_movement[
    ["nfl_id", "display_name", "draft_year", TARGET]
    + drill_predictors
].copy()

baseline_model.to_csv(OUT_BASELINE, index=False)
drill_model.to_csv(OUT_DRILLS, index=False)

print("\nBASELINE MOVEMENT MODEL")
print("Shape:", baseline_model.shape)
print("Predictors:", len(baseline_predictors))

print("\nMOVEMENT + DRILL MODEL")
print("Shape:", drill_model.shape)
print("Predictors:", len(drill_predictors))

print("\nTarget distribution:")
print(baseline_model[TARGET].value_counts())

print("\nSaved:")
print(OUT_BASELINE)
print(OUT_DRILLS)