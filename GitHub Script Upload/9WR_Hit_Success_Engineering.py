# NFL BIG DATA BOWL 2027
# WR Target Variable Developement

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
SOURCE = ROOT / "Source Data"
OUTPUT = ROOT / "Output Data"

integrated_file = OUTPUT / "WR_Integrated_Features.csv"
player_play_file = SOURCE / "player_play.csv"
games_file = SOURCE / "games.csv"

output_file = OUTPUT / "WR_Hit_Bust_Labeled.csv"
receiving_output = OUTPUT / "WR_Receiving_Production.csv"


wr = pd.read_csv(integrated_file)
player_play = pd.read_csv(player_play_file)
games = pd.read_csv(games_file)

print("WRs:", len(wr))
print("Player plays:", len(player_play))


player_play = player_play.merge(
    games[["game_id", "season"]],
    on="game_id",
    how="left"
)

wr_ids = wr["nfl_id"].unique()

player_play = player_play[
    player_play["nfl_id"].isin(wr_ids)
].copy()

player_play["target"] = (
    player_play["target"].fillna(0).astype(int)
)

player_play["rec_yards"] = (
    player_play["rec_yards"].fillna(0)
)

player_play["yards_after_catch"] = (
    player_play["yards_after_catch"].fillna(0)
)

# Aggregate player-season production
receiving = (
    player_play.groupby(["nfl_id", "season"])
    .agg(
        targets=("target", "sum"),
        receiving_yards=("rec_yards", "sum"),
        yards_after_catch=("yards_after_catch", "sum"),
        receiving_touchdowns=("play_description",
            lambda x: 0)
    )
    .reset_index()
)


receiving["receptions"] = (
    player_play.groupby(["nfl_id", "season"])["rec_yards"]
    .apply(lambda x: x.notna().sum())
    .values
)

receiving["yards_per_reception"] = (
    receiving["receiving_yards"] /
    receiving["receptions"].replace(0, np.nan)
)

receiving.to_csv(receiving_output, index=False)

print("\nReceiving production saved:", receiving_output)
print("Player-season rows:", len(receiving))

# FIRST TWO NFL SEASONS
# 2023/2024: first two NFL seasons
# 2025: first available NFL season only

receiving = receiving.merge(
    wr[["nfl_id", "draft_year"]],
    on="nfl_id",
    how="left"
)

receiving["evaluation_season"] = (
    receiving["season"] - receiving["draft_year"]
)

receiving = receiving[
    receiving["evaluation_season"].isin([0, 1])
].copy()

receiving_2yr = (
    receiving.groupby("nfl_id")
    .agg(
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        yards_after_catch=("yards_after_catch", "sum"),
        receiving_touchdowns=("receiving_touchdowns", "sum")
    )
    .reset_index()
)

df = wr.merge(
    receiving_2yr,
    on="nfl_id",
    how="left"
)

production_columns = [
    "targets",
    "receptions",
    "receiving_yards",
    "yards_after_catch",
    "receiving_touchdowns"
]

df[production_columns] = (
    df[production_columns].fillna(0)
)


thresholds = {
    1: (20, 2000, 1500),
    2: (15, 1500, 1000),
    3: (10, 1000, 750),
    4: (8, 750, 500),
    5: (5, 500, 400),
    6: (3, 300, 300),
    7: (2, 200, 200)
}

thresholds_2025 = {
    1: (10, 1000, 750),
    2: (8, 750, 500),
    3: (5, 500, 400),
    4: (4, 400, 300),
    5: (3, 300, 200),
    6: (2, 200, 150),
    7: (1, 100, 100)
}


def classify_player(row):

    round_number = row["draft_round"]
    draft_year = row["draft_year"]

    if pd.isna(round_number):
        return np.nan

    round_number = int(round_number)

    if round_number not in thresholds:
        return np.nan

    if draft_year == 2025:
        start_threshold, snap_threshold, yard_threshold = (
            thresholds_2025[round_number]
        )
    else:
        start_threshold, snap_threshold, yard_threshold = (
            thresholds[round_number]
        )

    starts_hit = (
        row["career_games_started"] >= start_threshold
    )

    snaps_hit = (
        row["career_offensive_snaps"] >= snap_threshold
    )

    yards_hit = (
        row["receiving_yards"] >= yard_threshold
    )

    # Must meet at least 2 of 3 criteria
    # Including either starts or receiving yards
    score = sum([starts_hit, snaps_hit, yards_hit])

    return int(
        score >= 2 and (starts_hit or yards_hit)
    )

df["hit_bust_label"] = df.apply(
    classify_player,
    axis=1
)

df["hit_bust"] = df["hit_bust_label"].map({
    1: "Hit",
    0: "Bust"
})


df["model_eligible"] = df["draft_year"].isin([2023, 2024])



print(df["hit_bust"].value_counts(dropna=False))

print("\nHIT / BUST BY DRAFT YEAR")
print(
    pd.crosstab(
        df["draft_year"],
        df["hit_bust"],
        margins=True
    )
)

print("\nHIT / BUST BY DRAFT ROUND")
print(
    pd.crosstab(
        df["draft_round"],
        df["hit_bust"],
        margins=True
    )
)


df.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Rows:", len(df))
print("Columns:", len(df.columns))

#even though I didn't end up using this I'm still going to keep it since I thought it was a cool idea