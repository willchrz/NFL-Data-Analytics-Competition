# 01_WR_Model_Data.py
# NFL BIG DATA BOWL 2027 - WR MODEL DATA PREPARATION

import pandas as pd
import numpy as np
from pathlib import Path

# 1. PATHS
ROOT = Path(r"W:\nfl-big-data-bowl-2027")
SRC = ROOT / "Source Data"
OUT = ROOT / "Output Data"

OUT.mkdir(exist_ok=True)

# LOAD WR PLAYERS AND COMBINE DATA
players = pd.read_csv(SRC / "players.csv")

wr = players[
    players["nfl_position"].astype(str).str.upper() == "WR"
].copy()

combine = pd.read_csv(SRC / "combine_results.csv")

combine = combine.merge(
    wr[["nfl_id", "draft_year"]],
    on=["nfl_id", "draft_year"],
    how="inner"
)

print("WR ROSTER:", wr.shape)
print("WR COMBINE:", combine.shape)

# COMBINE TRACKING PREPARATION
tracking = pd.read_csv(SRC / "combine_tracking.csv")

tracking = tracking[
    tracking["entity_type"].astype(str).str.upper() == "PLAYER"
].copy()

tracking = tracking[
    tracking["nfl_id"].isin(wr["nfl_id"])
].copy()

# Convert tracking time to numeric
tracking["time"] = pd.to_numeric(
    tracking["time"],
    errors="coerce"
)

tracking = tracking.dropna(subset=["time"])

tracking = tracking.sort_values(
    ["nfl_id", "event_id", "time"]
)

# Time differences within each drill attempt
tracking["dt"] = (
    tracking.groupby(["nfl_id", "event_id"])["time"]
    .diff()
)

# Longitudinal acceleration using speed changes over time
tracking["long_acc"] = (
    tracking.groupby(["nfl_id", "event_id"])["s"]
    .diff() / tracking["dt"]
)

tracking["long_acc"] = tracking["long_acc"].replace(
    [np.inf, -np.inf], np.nan
)

# Elapsed time within each event
tracking["elapsed"] = (
    tracking.groupby(["nfl_id", "event_id"])["time"]
    .transform(lambda x: x - x.min())
)

# FIRST-SECOND MOVEMENT FEATURES
first_second = tracking[
    tracking["elapsed"] <= 1
].copy()

first_features = first_second.groupby(
    ["nfl_id", "event_id"]
).agg(
    first_second_average_speed=("s", "mean"),
    first_second_maximum_speed=("s", "max"),
    first_second_average_acceleration=("a", "mean")
).reset_index()

# EVENT-LEVEL MOVEMENT FEATURES
events = tracking.groupby(
    ["nfl_id", "event_id"]
).agg(
    event_duration=("elapsed", "max"),
    average_speed=("s", "mean"),
    maximum_speed=("s", "max"),
    average_acceleration=("a", "mean"),
    maximum_acceleration=("a", "max"),
    maximum_longitudinal_acceleration=("long_acc", "max"),
    minimum_longitudinal_acceleration=("long_acc", "min")
).reset_index()

# Time to peak speed
peak = tracking.loc[
    tracking.groupby(["nfl_id", "event_id"])["s"].idxmax(),
    ["nfl_id", "event_id", "elapsed"]
].rename(
    columns={"elapsed": "time_to_peak_speed"}
)

events = events.merge(
    peak,
    on=["nfl_id", "event_id"],
    how="left"
)

events = events.merge(
    first_features,
    on=["nfl_id", "event_id"],
    how="left"
)

# PLAYER-LEVEL MOVEMENT AGGREGATION
movement = events.groupby("nfl_id").agg(
    total_events=("event_id", "count"),
    average_event_speed=("average_speed", "mean"),
    maximum_event_speed=("maximum_speed", "max"),
    average_event_acceleration=("average_acceleration", "mean"),
    maximum_event_acceleration=("maximum_acceleration", "max"),
    average_longitudinal_acceleration=(
        "maximum_longitudinal_acceleration", "mean"
    ),
    maximum_longitudinal_acceleration=(
        "maximum_longitudinal_acceleration", "max"
    ),
    minimum_longitudinal_acceleration=(
        "minimum_longitudinal_acceleration", "min"
    ),
    average_time_to_peak_speed=("time_to_peak_speed", "mean"),
    average_first_second_speed=("first_second_average_speed", "mean"),
    maximum_first_second_speed=("first_second_maximum_speed", "max"),
    average_first_second_acceleration=(
        "first_second_average_acceleration", "mean"
    ),
    average_event_duration=("event_duration", "mean"),
    maximum_event_duration=("event_duration", "max"),
    speed_variability=("average_speed", "std"),
    acceleration_variability=("average_acceleration", "std")
).reset_index()

# RAW TRACKING AGGREGATES
raw = tracking.groupby("nfl_id").agg(
    tracking_observations=("s", "count"),
    average_speed=("s", "mean"),
    maximum_speed=("s", "max"),
    speed_std=("s", "std"),
    average_raw_acceleration=("a", "mean"),
    maximum_raw_acceleration=("a", "max"),
    average_longitudinal_acceleration_movement=("long_acc", "mean"),
    maximum_longitudinal_acceleration_movement=("long_acc", "max"),
    minimum_longitudinal_acceleration_movement=("long_acc", "min")
).reset_index()

movement = movement.merge(
    raw,
    on="nfl_id",
    how="left"
)

# ORIGINAL MOVEMENT RATIO FEATURES
movement["acceleration_to_speed"] = (
    movement["average_raw_acceleration"] /
    movement["average_speed"].replace(0, np.nan)
)

movement["peak_to_average_speed_ratio"] = (
    movement["maximum_speed"] /
    movement["average_speed"].replace(0, np.nan)
)

movement["speed_variability_ratio"] = (
    movement["speed_std"] /
    movement["average_speed"].replace(0, np.nan)
)

movement["longitudinal_acceleration_range"] = (
    movement["maximum_longitudinal_acceleration_movement"] -
    movement["minimum_longitudinal_acceleration_movement"]
)

# INTEGRATE PLAYER + COMBINE + MOVEMENT
df = wr.merge(
    combine,
    on=["nfl_id", "draft_year"],
    how="left"
)

df = df.merge(
    movement,
    on="nfl_id",
    how="left"
)

# NFL RECEIVING PRODUCTION
plays = pd.read_csv(SRC / "player_play.csv")

games = pd.read_csv(SRC / "games.csv")[
    ["game_id", "season"]
]

plays = plays.merge(
    games,
    on="game_id",
    how="left"
)

plays = plays[
    plays["nfl_id"].isin(wr["nfl_id"])
].copy()

plays["target"] = (
    plays["target"].fillna(0).astype(int)
)

plays["rec_yards"] = plays["rec_yards"].fillna(0)

plays["yards_after_catch"] = (
    plays["yards_after_catch"].fillna(0)
)

# First two NFL seasons after each player's draft
plays = plays.merge(
    wr[["nfl_id", "draft_year"]],
    on="nfl_id",
    how="left"
)

plays = plays[
    (plays["season"] >= plays["draft_year"]) &
    (plays["season"] <= plays["draft_year"] + 1)
].copy()

receiving = plays.groupby("nfl_id").agg(
    targets=("target", "sum"),
    receiving_yards=("rec_yards", "sum"),
    yards_after_catch=("yards_after_catch", "sum")
).reset_index()

df = df.merge(
    receiving,
    on="nfl_id",
    how="left"
)

df[
    ["targets", "receptions", "receiving_yards", "yards_after_catch"]
] = df[
    ["targets", "receptions", "receiving_yards", "yards_after_catch"]
].fillna(0)

# CAREER SUCCESS DATA
career = pd.read_csv(
    SRC / "player_career_successes.csv"
)

df = df.merge(
    career,
    on="nfl_id",
    how="left"
)

# RELATIVE HIT / BUST LABEL
# Standardize career success metrics within draft class
success_metrics = [
    "career_receiving_yards",
    "career_starts",
    "career_offensive_snaps",
    "career_receptions"
]

# Adjust these names if the career CSV uses different column names.
for col in success_metrics:
    df[col + "_z"] = (
        df.groupby("draft_year")[col]
        .transform(
            lambda x: (x - x.mean()) / x.std()
        )
    )

df["relative_success_score"] = (
    0.40 * df["career_receiving_yards_z"] +
    0.25 * df["career_starts_z"] +
    0.20 * df["career_offensive_snaps_z"] +
    0.15 * df["career_receptions_z"]
)

# Bottom 40% of each draft class = Bust
df["target"] = (
    df.groupby("draft_year")["relative_success_score"]
    .transform(
        lambda x: (x <= x.quantile(0.40)).astype(int)
    )
)

# Convert to desired encoding:
# 0 = Bust, 1 = Hit
df["target"] = 1 - df["target"]

# RESTRICT MODELING TO 2023-2024
df = df[
    df["draft_year"].isin([2023, 2024])
].copy()

# SAVE MODEL DATASET
output_file = OUT / "WR_Model_Dataset.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("MODEL DATASET CREATED")
print("=" * 60)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Hits:", int(df["target"].sum()))
print("Busts:", int((df["target"] == 0).sum()))
print("Saved:", output_file)
