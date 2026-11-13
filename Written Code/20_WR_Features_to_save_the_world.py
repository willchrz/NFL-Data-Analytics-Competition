# NFL BIG DATA BOWL 2027
# Whats scarier than a ghost? Feature engineering! WR Additional features

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

final_movement = pd.read_csv(
    OUTPUT / "WR_Final_Movement_Features.csv"
)

advanced_events = pd.read_csv(
    OUTPUT / "WR_Advanced_Event_Features.csv"
)

event_summary = pd.read_csv(
    OUTPUT / "WR_Combine_Event_Summary.csv"
)

print("\nMOVEMENT FEATURES")
print(final_movement.columns.tolist())

print("\nADVANCED EVENT FEATURES")
print(advanced_events.columns.tolist())

print("\nEVENT SUMMARY FEATURES")
print(event_summary.columns.tolist())

if "drill_type" in advanced_events.columns:
    print("\nDRILL TYPES")
    print(advanced_events["drill_type"].value_counts())

elif "drill_name" in advanced_events.columns:
    print("\nDRILL NAMES")
    print(advanced_events["drill_name"].value_counts())

print("\nFINAL MOVEMENT SAMPLE")
print(final_movement.head())

print("\nADVANCED EVENT SAMPLE")
print(advanced_events.head())

print("\nEVENT SUMMARY SAMPLE")
print(event_summary.head())

print("\nSCRIPT 20 INSPECTION COMPLETED")
