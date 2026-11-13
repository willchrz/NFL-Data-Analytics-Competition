# NFL BIG DATA BOWL 2027
# The last models we're not good enough so I'm back here again :)

import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
INPUT_FILE = ROOT / "Output Data" / "WR_Model_Relative.csv"
OUTPUT_FILE = ROOT / "Output Data" / "WR_XGBoost_Results.csv"
IMPORTANCE_FILE = ROOT / "Output Data" / "WR_XGBoost_Feature_Importance.csv"

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["target", "nfl_id"], errors="ignore")
X = X.select_dtypes(include="number")

y = df["target"]

print("\nWR XGBOOST MODEL")
print("Players:", len(df))
print("Predictors:", X.shape[1])

print("\nClass Distribution:")
print(y.value_counts())

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("xgboost", XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ))
])

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

print("\nXGBOOST CROSS-VALIDATION RESULTS")

for metric in scoring:
    scores = results[f"test_{metric}"]

    print(
        f"{metric.upper()}: "
        f"{scores.mean():.4f} "
        f"(SD: {scores.std():.4f})"
    )

model.fit(X, y)

xgb = model.named_steps["xgboost"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance["Rank"] = range(1, len(importance) + 1)

importance = importance[
    ["Rank", "Feature", "Importance"]
]

print("\nTOP 15 XGBOOST FEATURES")

print(
    importance.head(15).to_string(index=False)
)

summary = pd.DataFrame({
    "Model": ["XGBoost"],
    "CV_Accuracy": [results["test_accuracy"].mean()],
    "CV_Precision": [results["test_precision"].mean()],
    "CV_Recall": [results["test_recall"].mean()],
    "CV_F1": [results["test_f1"].mean()],
    "CV_ROC_AUC": [results["test_roc_auc"].mean()]
})

summary.to_csv(OUTPUT_FILE, index=False)

importance.to_csv(IMPORTANCE_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("Saved:", IMPORTANCE_FILE)