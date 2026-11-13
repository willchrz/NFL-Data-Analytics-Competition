
# ============================================================
# Script 11: WR Movement Outlier Investigation THIS ONE IS NOT INTEGRAL WHATSOEVER DO WITH IT WHAT YOU WANT BUT IT WILL NOT AID IN RECREATING STUDY
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1. Define project folders
# ------------------------------------------------------------

project_folder = Path(r"W:\nfl-big-data-bowl-2027")

source_folder = project_folder / "Source Data"
output_folder = project_folder / "Output Data"

# ------------------------------------------------------------
# 2. Load datasets
# ------------------------------------------------------------

outliers = pd.read_csv(
    output_folder / "WR_Movement_Outlier_Report.csv"
)

players = pd.read_csv(
    output_folder / "WR_Player_List.csv"
)

tracking = pd.read_csv(
    source_folder / "combine_tracking.csv"
)

print("\nWR MOVEMENT OUTLIER INVESTIGATION")

# ------------------------------------------------------------
# 3. Identify flagged WRs
# ------------------------------------------------------------

flagged_ids = outliers["nfl_id"].unique()

print("Flagged WRs:", len(flagged_ids))
print("Flagged metric records:", len(outliers))

# ------------------------------------------------------------
# 4. Filter raw tracking to flagged WRs
# ------------------------------------------------------------

tracking_wr = tracking[
    tracking["nfl_id"].isin(flagged_ids)
].copy()

tracking_wr = tracking_wr.merge(
    players[["nfl_id", "display_name"]],
    on="nfl_id",
    how="left"
)

tracking_wr["time"] = pd.to_datetime(
    tracking_wr["time"]
)

tracking_wr = tracking_wr.sort_values(
    ["nfl_id", "event_id", "time"]
)

print("Raw tracking observations:", len(tracking_wr))

# ------------------------------------------------------------
# 5. Create event-level movement summaries
# ------------------------------------------------------------

tracking_wr["elapsed_seconds"] = (
    tracking_wr.groupby(
        ["nfl_id", "event_id"]
    )["time"]
    .transform(lambda x: (x - x.min()).dt.total_seconds())
)

event_summary = (
    tracking_wr
    .groupby(
        [
            "nfl_id",
            "display_name",
            "draft_year",
            "event_id",
            "drill_name",
            "attempt"
        ],
        as_index=False
    )
    .agg(
        observations=("time", "count"),
        duration=("elapsed_seconds", "max"),
        maximum_speed=("s", "max"),
        maximum_acceleration=("a", "max"),
        average_speed=("s", "mean"),
        average_acceleration=("a", "mean")
    )
)

event_summary.to_csv(
    output_folder / "WR_Outlier_Event_Investigation.csv",
    index=False
)

# ------------------------------------------------------------
# 6. Find peak speed and acceleration observations
# ------------------------------------------------------------

peak_speed_rows = (
    tracking_wr.loc[
        tracking_wr.groupby("nfl_id")["s"].idxmax()
    ]
    .copy()
)

peak_speed_rows["peak_metric"] = "maximum_raw_speed"

peak_acceleration_rows = (
    tracking_wr.loc[
        tracking_wr.groupby("nfl_id")["a"].idxmax()
    ]
    .copy()
)

peak_acceleration_rows["peak_metric"] = (
    "maximum_raw_acceleration"
)

peak_observations = pd.concat(
    [peak_speed_rows, peak_acceleration_rows],
    ignore_index=True
)

peak_observations.to_csv(
    output_folder / "WR_Outlier_Peak_Observations.csv",
    index=False
)

print("\nPEAK RAW OBSERVATIONS")
print(
    peak_observations[
        [
            "display_name",
            "draft_year",
            "drill_name",
            "event_id",
            "time",
            "s",
            "a",
            "peak_metric"
        ]
    ].head(20)
)

# ------------------------------------------------------------
# 7. Extract tracking sequences around peak observations
# ------------------------------------------------------------

sequence_records = []

for _, peak in peak_observations.iterrows():

    player_id = peak["nfl_id"]
    event_id = peak["event_id"]
    peak_time = peak["time"]

    event_tracking = tracking_wr[
        (tracking_wr["nfl_id"] == player_id)
        & (tracking_wr["event_id"] == event_id)
    ].copy()

    event_tracking["seconds_from_peak"] = (
        event_tracking["time"] - peak_time
    ).dt.total_seconds()

    sequence = event_tracking[
        event_tracking["seconds_from_peak"].between(
            -0.5, 0.5
        )
    ].copy()

    sequence["peak_metric"] = peak["peak_metric"]

    sequence_records.append(sequence)

if sequence_records:

    peak_sequences = pd.concat(
        sequence_records,
        ignore_index=True
    )

else:

    peak_sequences = pd.DataFrame()

peak_sequences.to_csv(
    output_folder / "WR_Outlier_Tracking_Sequences.csv",
    index=False
)

# ------------------------------------------------------------
# 8. Print highest event-level acceleration values
# ------------------------------------------------------------

print("\nTOP 20 EVENT ACCELERATION VALUES")

print(
    event_summary.sort_values(
        "maximum_acceleration",
        ascending=False
    )[
        [
            "display_name",
            "draft_year",
            "drill_name",
            "attempt",
            "observations",
            "duration",
            "maximum_speed",
            "maximum_acceleration"
        ]
    ].head(20)
)

# ------------------------------------------------------------
# 9. Print highest event-level speeds
# ------------------------------------------------------------

print("\nTOP 20 EVENT SPEED VALUES")

print(
    event_summary.sort_values(
        "maximum_speed",
        ascending=False
    )[
        [
            "display_name",
            "draft_year",
            "drill_name",
            "attempt",
            "observations",
            "duration",
            "maximum_speed",
            "maximum_acceleration"
        ]
    ].head(20)
)

# ------------------------------------------------------------
# 10. Save investigation summary
# ------------------------------------------------------------

print("\nINVESTIGATION OUTPUTS")

print(
    "Event investigation:",
    output_folder / "WR_Outlier_Event_Investigation.csv"
)

print(
    "Peak observations:",
    output_folder / "WR_Outlier_Peak_Observations.csv"
)

print(
    "Tracking sequences:",
    output_folder / "WR_Outlier_Tracking_Sequences.csv"
)

print("\nOutlier investigation completed successfully.")
