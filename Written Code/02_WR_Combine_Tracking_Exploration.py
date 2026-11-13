# NFL BIG DATA BOWL 2027
# WR COMBINE TRACKING EXPLORATION


import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1. Define directories
# ------------------------------------------------------------

PROJECT_DIR = Path(r"W:\nfl-big-data-bowl-2027")

SOURCE_DIR = PROJECT_DIR / "Source Data"

OUTPUT_DIR = PROJECT_DIR / "Output Data"

# ------------------------------------------------------------
# 2. Load datasets
# ------------------------------------------------------------

wr_players = pd.read_csv(
    OUTPUT_DIR / "WR_Player_List.csv"
)

combine_tracking = pd.read_csv(
    SOURCE_DIR / "combine_tracking.csv"
)

# ------------------------------------------------------------
# 3. Filter tracking data to WRs
# ------------------------------------------------------------

wr_tracking = combine_tracking[
    combine_tracking["nfl_id"].isin(wr_players["nfl_id"])
].copy()

print("\nWR COMBINE TRACKING")

print("Total tracking observations:", len(wr_tracking))

print("Unique WRs in tracking:", wr_tracking["nfl_id"].nunique())

# ------------------------------------------------------------
# 4. Examine available drills
# ------------------------------------------------------------

print("\nDrill types:")

print(
    wr_tracking["drill_type"]
    .value_counts()
)

print("\nDrill names:")

print(
    wr_tracking["drill_name"]
    .value_counts()
)

# ------------------------------------------------------------
# 5. Count tracking observations per player
# ------------------------------------------------------------

player_tracking_counts = (
    wr_tracking
    .groupby(["nfl_id", "draft_year"])
    .size()
    .reset_index(name="tracking_observations")
)

player_tracking_counts = player_tracking_counts.merge(
    wr_players[["nfl_id", "display_name"]],
    on="nfl_id",
    how="left"
)

player_tracking_counts.to_csv(
    OUTPUT_DIR / "WR_Combine_Tracking_Counts.csv",
    index=False
)

# ------------------------------------------------------------
# 6. Count observations per player and drill
# ------------------------------------------------------------

player_drill_counts = (
    wr_tracking
    .groupby(["nfl_id", "draft_year", "drill_name"])
    .size()
    .reset_index(name="observations")
)

player_drill_counts = player_drill_counts.merge(
    wr_players[["nfl_id", "display_name"]],
    on="nfl_id",
    how="left"
)

player_drill_counts.to_csv(
    OUTPUT_DIR / "WR_Combine_Drill_Counts.csv",
    index=False
)

# ------------------------------------------------------------
# 7. Inspect acceleration and speed
# ------------------------------------------------------------

print("\nAcceleration summary:")

print(
    wr_tracking["a"].describe()
)

print("\nSpeed summary:")

print(
    wr_tracking["s"].describe()
)

print("\nExploration complete.")
