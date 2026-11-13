# NFL BIG DATA BOWL 2027
# WR first round of modelling datasets because I'm way too optimistic

import pandas as pd
from pathlib import Path

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

input_file = OUTPUT / "WR_Dual_Success_Labels.csv"

absolute_output = OUTPUT / "WR_Model_Absolute.csv"
relative_output = OUTPUT / "WR_Model_Relative.csv"

df = pd.read_csv(input_file)

# Use established draft classes for modeling
df = df[df["draft_year"].isin([2023, 2024])].copy()

print("2023-2024 WRs:", len(df))

excluded_columns = [
    "nfl_id",
    "display_name",
    "draft_year",
    "nfl_position",
    "birth_date",
    "college_name",
    "college_conference",
    "draft_round",
    "draft_pick_within_round",
    "draft_overall_pick",
    "combine_position",

    # Career outcomes
    "career_offensive_snaps",
    "career_defensive_snaps",
    "career_special_teams_snaps",
    "career_games_active",
    "career_games_started",
    "ap_all_pro_1st_team",
    "ap_all_pro_2nd_team",
    "pro_bowl_original_ballot",

    # Receiving outcomes
    "targets",
    "receptions",
    "receiving_yards",
    "yards_after_catch",
    "receiving_touchdowns",
    "yards_per_reception",

    # Absolute label fields
    "hit_bust_label",
    "hit_bust",
    "model_eligible",

    # Relative label fields
    "relative_success_score",
    "relative_rank",
    "draft_class_size",
    "relative_percentile",
    "relative_hit_bust_label",
    "relative_hit_bust",

    # Outcome calculation components
    "receiving_yards_z",
    "career_games_started_z",
    "career_offensive_snaps_z",
    "receptions_z"
]

X = df.drop(
    columns=excluded_columns,
    errors="ignore"
)

# Numeric predictors only
X = X.select_dtypes(include="number")

# Remove completely empty columns
X = X.dropna(axis=1, how="all")

absolute_mask = df["hit_bust_label"].notna()

absolute_df = X.loc[absolute_mask].copy()

absolute_df["nfl_id"] = df.loc[absolute_mask, "nfl_id"].values
absolute_df["display_name"] = (
    df.loc[absolute_mask, "display_name"].values
)
absolute_df["draft_year"] = (
    df.loc[absolute_mask, "draft_year"].values
)
absolute_df["target"] = (
    df.loc[absolute_mask, "hit_bust_label"].astype(int).values
)

absolute_df.to_csv(absolute_output, index=False)

relative_df = X.copy()

relative_df["nfl_id"] = df["nfl_id"].values
relative_df["display_name"] = df["display_name"].values
relative_df["draft_year"] = df["draft_year"].values
relative_df["target"] = (
    df["relative_hit_bust_label"].astype(int).values
)

relative_df.to_csv(relative_output, index=False)

print("\nPredictor columns:")
print(X.columns.tolist())

print("\nSaved:")
print(absolute_output)
print(relative_output)