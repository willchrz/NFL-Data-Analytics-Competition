
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUT = ROOT / "Output Data"
OUT.mkdir(exist_ok=True)

# LOAD DATA
df = pd.read_csv(OUT / "WR_Model_Dataset.csv")

X = df.select_dtypes("number").drop(
    columns=["nfl_id", "draft_year", "target"], errors="ignore"
)
y = df["target"].astype(int)

# MODELS
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
predictions = []
feature_counts = {}

# OUT-OF-FOLD VALIDATION
for name, model in models.items():

    oof_prob = np.zeros(len(df))
    oof_pred = np.zeros(len(df), dtype=int)

    for train_idx, test_idx in cv.split(X, y):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        imputer = SimpleImputer(strategy="median")

        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        selector = SelectKBest(f_classif, k=15)
        X_train_sel = selector.fit_transform(X_train_imp, y_train)
        X_test_sel = selector.transform(X_test_imp)

        selected = X.columns[selector.get_support()]

        for feature in selected:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

        model.fit(X_train_sel, y_train)

        oof_prob[test_idx] = model.predict_proba(X_test_sel)[:, 1]
        oof_pred[test_idx] = model.predict(X_test_sel)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y, oof_pred),
        "Precision": precision_score(y, oof_pred),
        "Recall": recall_score(y, oof_pred),
        "F1": f1_score(y, oof_pred),
        "AUC": roc_auc_score(y, oof_prob)
    })

    for i in range(len(df)):
        predictions.append({
            "nfl_id": df.iloc[i].nfl_id,
            "display_name": df.iloc[i].display_name,
            "draft_year": df.iloc[i].draft_year,
            "model": name,
            "actual": y.iloc[i],
            "predicted": oof_pred[i],
            "probability_hit": oof_prob[i],
            "predicted_label": (
                "Hit" if oof_pred[i] == 1 else "Bust"
            )
        })

    # TRAIN FINAL MODEL ON ALL PLAYERS
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X_imp, y)

    model.fit(X_selected, y)

    joblib.dump(
        {
            "imputer": imputer,
            "selector": selector,
            "model": model,
            "features": list(X.columns)
        },
        OUT / f"WR_{name.replace(' ', '_')}_Final.pkl"
    )

# SAVE OUTPUTS
pd.DataFrame(results).to_csv(
    OUT / "WR_Final_Model_Comparison.csv", index=False
)

pd.DataFrame(predictions).to_csv(
    OUT / "WR_Final_OutOfFold_Predictions.csv", index=False
)

for name in models:
    pred = pd.DataFrame(predictions)
    pred = pred[pred.model == name]

    pd.DataFrame(
        confusion_matrix(pred.actual, pred.predicted),
        index=["Actual Bust", "Actual Hit"],
        columns=["Predicted Bust", "Predicted Hit"]
    ).to_csv(
        OUT / f"WR_{name.replace(' ', '_')}_Confusion_Matrix.csv"
    )

pd.DataFrame(
    feature_counts.items(), columns=["Feature", "Selection_Count"]
).sort_values("Selection_Count", ascending=False).to_csv(
    OUT / "WR_Final_Feature_Selection.csv", index=False
)

print("\nFINAL MODEL PIPELINE COMPLETED")
print(pd.DataFrame(results).round(3))
print(f"\nSaved outputs to: {OUT}")
