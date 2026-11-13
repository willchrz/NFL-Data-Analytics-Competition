# NFL BIG DATA BOWL 2027
# The first WR variable was NOT good so we gotta make another one
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

input_file = OUTPUT / "WR_Hit_Bust_Labeled.csv"
output_file = OUTPUT / "WR_Dual_Success_Labels.csv"

df = pd.read_csv(input_file)

print("\nWR RELATIVE SUCCESS ENGINEERING")
print("Original rows:", len(df))

performance_metrics = [
    "receiving_yards",
    "career_games_started",
    "career_offensive_snaps",
    "receptions"
]

# STANDARDIZE PERFORMANCE WITHIN DRAFT CLASS
for metric in performance_metrics:

    # Z-score standardization within draft class
    df[f"{metric}_z"] = (
        df.groupby("draft_year")[metric]
        .transform(
            lambda x: (
                x - x.mean()
            ) / x.std()
        )
    )


df["relative_success_score"] = (
    0.40 * df["receiving_yards_z"] +
    0.25 * df["career_games_started_z"] +
    0.20 * df["career_offensive_snaps_z"] +
    0.15 * df["receptions_z"]
)

df["relative_rank"] = (
    df.groupby("draft_year")["relative_success_score"]
    .rank(method="first", ascending=False)
)

df["draft_class_size"] = (
    df.groupby("draft_year")["relative_success_score"]
    .transform("count")
)

df["relative_percentile"] = (
    df.groupby("draft_year")["relative_success_score"]
    .rank(method="average", pct=True)
)

df["relative_hit_bust_label"] = np.where(
    df["relative_percentile"] <= 0.40,
    0,
    1
)

df["relative_hit_bust"] = df[
    "relative_hit_bust_label"
].map({
    1: "Relative Hit",
    0: "Relative Bust"
})


print("\nABSOLUTE HIT/BUST")
print(df["hit_bust"].value_counts(dropna=False))

print("\nRELATIVE HIT/BUST")
print(
    df["relative_hit_bust"]
    .value_counts(dropna=False)
)

print("\nRELATIVE LABELS BY DRAFT YEAR")
print(
    pd.crosstab(
        df["draft_year"],
        df["relative_hit_bust"],
        margins=True
    )
)

print("\nRELATIVE SCORE SUMMARY")
print(
    df.groupby("draft_year")["relative_success_score"]
    .describe()
)


df.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Rows:", len(df))
print("Columns:", len(df.columns))
