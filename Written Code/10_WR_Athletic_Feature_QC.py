# WR Athletic Feature Quality Control

import pandas as pd
import numpy as np
from pathlib import Path

project_folder = Path(r"W:\nfl-big-data-bowl-2027")
output_folder = project_folder / "Output Data"

df = pd.read_csv(
    output_folder / "WR_Combine_Athletic_Profile.csv"
)

print("Total rows:", len(df))
print("Unique NFL IDs:", df["nfl_id"].nunique())
print("Duplicate NFL IDs:", df["nfl_id"].duplicated().sum())

missing_values = (
    df.isna()
    .sum()
    .reset_index()
)

missing_values.columns = [
    "variable",
    "missing_count"
]

missing_values["missing_percentage"] = (
    missing_values["missing_count"] / len(df) * 100
)

missing_values = missing_values.sort_values(
    "missing_percentage",
    ascending=False
)

missing_values.to_csv(
    output_folder / "WR_Athletic_Missingness_Report.csv",
    index=False
)

print("\nMissing Valuesd")
print(missing_values.head(15))

combine_variables = [
    "forty",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
    "combine_weight",
    "combine_height",
    "arm_length",
    "wing_span",
    "hand_size"
]

combine_summary = df[combine_variables].describe().T

combine_summary.to_csv(
    output_folder / "WR_Combine_Statistics.csv"
)

print("\nCOMBINE SUMMARY")
print(combine_summary)

movement_variables = [
    "average_event_speed",
    "maximum_event_speed",
    "average_event_acceleration",
    "maximum_event_acceleration",
    "average_first_second_speed",
    "maximum_first_second_speed",
    "average_first_second_acceleration",
    "average_time_to_peak_speed",
    "speed_variability",
    "acceleration_variability"
]

movement_summary = df[movement_variables].describe().T

movement_summary.to_csv(
    output_folder / "WR_Movement_Statistics.csv"
)

print("\nMOVEMENT SUMMARY")
print(movement_summary)

outlier_variables = [
    "maximum_event_speed",
    "maximum_event_acceleration",
    "maximum_first_second_speed",
    "average_first_second_acceleration"
]

outlier_records = []

for variable in outlier_variables:

    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[variable] < lower_bound)
        | (df[variable] > upper_bound)
    ]

    for _, row in outliers.iterrows():

        outlier_records.append({
            "nfl_id": row["nfl_id"],
            "display_name": row["display_name"],
            "draft_year": row["draft_year"],
            "variable": variable,
            "value": row[variable],
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        })

outlier_report = pd.DataFrame(outlier_records)

outlier_report.to_csv(
    output_folder / "WR_Movement_Outlier_Report.csv",
    index=False
)

print("\nMOVEMENT OUTLIER REPORT")
print("Potential outlier records:", len(outlier_report))

if len(outlier_report) > 0:
    print(outlier_report)

print("\nSUSPICIOUS VALUE CHECKS")

print(
    "Negative 40-yard dash times:",
    (df["forty"] < 0).sum()
)

print(
    "Negative vertical jumps:",
    (df["vertical"] < 0).sum()
)

print(
    "Negative broad jumps:",
    (df["broad_jump"] < 0).sum()
)

print(
    "Maximum event speeds above 12 yd/s:",
    (df["maximum_event_speed"] > 12).sum()
)

print(
    "Maximum event accelerations above 20 yd/s²:",
    (df["maximum_event_acceleration"] > 20).sum()
)

qc_overview = pd.DataFrame({
    "check": [
        "Total WRs",
        "Unique NFL IDs",
        "Duplicate NFL IDs",
        "Potential movement outlier records",
        "WRs with missing 40-yard dash",
        "WRs with missing vertical jump",
        "WRs with missing broad jump",
        "WRs with missing three-cone",
        "WRs with missing short shuttle"
    ],
    "result": [
        len(df),
        df["nfl_id"].nunique(),
        df["nfl_id"].duplicated().sum(),
        len(outlier_report),
        df["forty"].isna().sum(),
        df["vertical"].isna().sum(),
        df["broad_jump"].isna().sum(),
        df["three_cone"].isna().sum(),
        df["short_shuttle"].isna().sum()
    ]
})

qc_overview.to_csv(
    output_folder / "WR_Athletic_QC_Overview.csv",
    index=False
)

print("\nQC OVERVIEW")
print(qc_overview)

print("\nQuality control completed successfully.")
