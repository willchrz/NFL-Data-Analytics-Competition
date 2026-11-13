
import pandas as pd
from pathlib import Path

# Project folders
ROOT = Path(r"W:\nfl-big-data-bowl-2027")
SOURCE = ROOT / "Source Data"
OUTPUT = ROOT / "Output Data"

# Input files
athletic_file = OUTPUT / "WR_Combine_Athletic_Profile.csv"
movement_file = OUTPUT / "WR_Final_Movement_Features.csv"
career_file = SOURCE / "player_career_successes.csv"

# Output file
output_file = OUTPUT / "WR_Integrated_Features.csv"

# Load datasets
athletic = pd.read_csv(athletic_file)
movement = pd.read_csv(movement_file)
career = pd.read_csv(career_file)

print("\nWR FEATURE INTEGRATION")
print("Athletic profile:", athletic.shape)
print("Movement features:", movement.shape)
print("Career outcomes:", career.shape)

print("\nCareer columns:")
print(career.columns.tolist())

# Merge athletic and movement features
integrated = athletic.merge(
    movement,
    on=["nfl_id", "display_name"],
    how="left",
    suffixes=("", "_movement")
)

# Merge career outcomes
integrated = integrated.merge(
    career,
    on="nfl_id",
    how="left"
)

# Save integrated dataset
integrated.to_csv(output_file, index=False)

print("\nINTEGRATED DATASET")
print("Rows:", len(integrated))
print("Columns:", len(integrated.columns))
print("Unique WRs:", integrated["nfl_id"].nunique())

print("\nSaved:", output_file)
print("\nSCRIPT 14 COMPLETED")