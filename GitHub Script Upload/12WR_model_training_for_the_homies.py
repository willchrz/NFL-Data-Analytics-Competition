# NFL BIG DATA BOWL 2027
# They fly now? First WR Model Training

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

absolute_file = OUTPUT / "WR_Model_Absolute.csv"
relative_file = OUTPUT / "WR_Model_Relative.csv"
results_file = OUTPUT / "WR_Model_Comparison.csv"

absolute_df = pd.read_csv(absolute_file)
relative_df = pd.read_csv(relative_file)

print("\nWR MODEL TRAINING")
print("Absolute dataset:", absolute_df.shape)
print("Relative dataset:", relative_df.shape)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ))
])

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

def evaluate_models(dataset, target_name):

    X = dataset.drop(
        columns=["nfl_id", "display_name", "draft_year", "target"]
    )

    y = dataset["target"]

    results = []

    print("\n" + "=" * 60)
    print(target_name)
    print("=" * 60)

    print("\nClass distribution:")
    print(y.value_counts())

    for model_name, model in models.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc"
            },
            return_train_score=False
        )

        model.fit(X, y)

        predictions = model.predict(X)

        print("\nMODEL:", model_name)

        print("\nCross-Validation Results:")
        print("Accuracy:", scores["test_accuracy"].mean())
        print("Precision:", scores["test_precision"].mean())
        print("Recall:", scores["test_recall"].mean())
        print("F1:", scores["test_f1"].mean())
        print("ROC-AUC:", scores["test_roc_auc"].mean())

        print("\nFull Dataset Confusion Matrix:")
        print(confusion_matrix(y, predictions))

        results.append({
            "target": target_name,
            "model": model_name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_precision": scores["test_precision"].mean(),
            "cv_recall": scores["test_recall"].mean(),
            "cv_f1": scores["test_f1"].mean(),
            "cv_roc_auc": scores["test_roc_auc"].mean()
        })

    return results

all_results = []

all_results.extend(
    evaluate_models(
        absolute_df,
        "Absolute Hit/Bust"
    )
)

all_results.extend(
    evaluate_models(
        relative_df,
        "Relative Hit/Bust"
    )
)

results_df = pd.DataFrame(all_results)

results_df.to_csv(results_file, index=False)

print("\nMODEL COMPARISON")
print(results_df)

print("\nSaved:", results_file)
print("\nSCRIPT 17 COMPLETED")
