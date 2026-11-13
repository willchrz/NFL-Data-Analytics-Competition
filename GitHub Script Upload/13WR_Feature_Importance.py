# NFL BIG DATA BOWL 2027
# WR Model 1 Feature Importance

import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(r"W:\nfl-big-data-bowl-2027")

INPUT_FILE = ROOT / "Output Data" / "WR_Model_Relative.csv"

OUTPUT_FILE = ROOT / "Output Data" / "WR_Feature_Importance_Clean.csv"

df = pd.read_csv(INPUT_FILE)

target = "target"

X = df.drop(columns=[target])

# Remove identifiers
X = X.drop(columns=["nfl_id"], errors="ignore")

# Numeric predictors only
X = X.select_dtypes(include="number")

y = df[target]

print("\nWR FEATURE IMPORTANCE - CLEAN VERSION")
print("Players:", len(df))
print("Predictors:", X.shape[1])

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X, y)

rf = model.named_steps["model"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance["Rank"] = range(1, len(importance) + 1)

importance = importance[
    ["Rank", "Feature", "Importance"]
]

print("\nTOP 15 FEATURES")

print(importance.head(15).to_string(index=False))

importance.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)