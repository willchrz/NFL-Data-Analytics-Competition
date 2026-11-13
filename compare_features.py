import pandas as pd

root = r"W:\nfl-big-data-bowl-2027\Output Data"

old = pd.read_csv(root + r"\WR_Model_MovementOnly.csv")
new = pd.read_csv(root + r"\WR_Model_Dataset.csv")

print("ORIGINAL MOVEMENT:", old.shape)
print("REBUILT DATASET:", new.shape)

old_features = set(old.columns) - {"nfl_id","display_name","draft_year","target"}
new_features = set(new.columns) - {"nfl_id","display_name","draft_year","target","ngs_college_production_score","ngs_athleticism_score","ngs_final_score"}

print("\nOLD FEATURES MISSING FROM REBUILD:")
print(sorted(old_features - new_features))

print("\nNEW FEATURES NOT IN ORIGINAL:")
print(sorted(new_features - old_features))

print("\nORIGINAL TARGET COUNTS:")
print(old["target"].value_counts().sort_index())

print("\nREBUILT TARGET COUNTS:")
print(new["target"].value_counts().sort_index())
