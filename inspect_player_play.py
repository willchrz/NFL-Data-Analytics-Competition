import pandas as pd

p = r"W:\nfl-big-data-bowl-2027\Source Data\player_play.csv"
df = pd.read_csv(p, nrows=5)

print("COLUMNS:")
print(df.columns.tolist())

print("\nSAMPLE:")
print(df.head())
