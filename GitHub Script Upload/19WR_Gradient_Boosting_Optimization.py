# NFL BIG DATA BOWL 2027
# XGBoost gotta save me

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

BASE_FILE = OUTPUT / "WR_Model_MovementOnly.csv"
RESULT_FILE = OUTPUT / "WR_Gradient_Boosting_Results.csv"

df = pd.read_csv(BASE_FILE)

TARGET = "target"

y = df[TARGET].astype(int)

metadata = [
    "nfl_id",
    "display_name",
    "draft_year",
    TARGET
]

traditional_features = [
    "combine_height",
    "combine_weight",
    "hand_size",
    "arm_length",
    "wing_span",
    "ten_yd_split",
    "forty",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
    "bench_reps"
]

traditional_features = [
    c for c in traditional_features
    if c in df.columns
]

movement_features = [
    c for c in df.columns
    if c not in metadata + traditional_features
    and pd.api.types.is_numeric_dtype(df[c])
]

feature_groups = {
    "Tracking Movement": movement_features,
    "Traditional Combine": traditional_features,
    "Traditional + Tracking":
        traditional_features + movement_features
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        eval_metric="logloss",
        random_state=42
    )
}

results = []

for group_name, features in feature_groups.items():

    X = df[features]

    print("\n" + "=" * 60)
    print(group_name)
    print("Original predictors:", len(features))
    print("=" * 60)

    for model_name, model in models.items():

        # Test no selection and three selected feature sizes
        feature_counts = [
            min(10, len(features)),
            min(15, len(features)),
            min(20, len(features)),
            len(features)
        ]

        feature_counts = list(dict.fromkeys(feature_counts))

        for k in feature_counts:

            steps = [
                ("imputer", SimpleImputer(strategy="median"))
            ]

            if model_name == "Gradient Boosting":
                steps.append(
                    ("selector", SelectKBest(
                        score_func=f_classif,
                        k=k
                    ))
                )

            elif model_name == "XGBoost":
                steps.append(
                    ("selector", SelectKBest(
                        score_func=f_classif,
                        k=k
                    ))
                )

            elif model_name == "Random Forest":
                steps.append(
                    ("selector", SelectKBest(
                        score_func=f_classif,
                        k=k
                    ))
                )

            steps.append(("model", model))

            pipeline = Pipeline(steps)

            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )

            row = {
                "Feature_Group": group_name,
                "Model": model_name,
                "Selected_Features": k,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1": scores["test_f1"].mean(),
                "ROC_AUC": scores["test_roc_auc"].mean()
            }

            results.append(row)

            print(
                f"{model_name} | Features={k} | "
                f"Accuracy={row['Accuracy']:.3f} | "
                f"F1={row['F1']:.3f} | "
                f"AUC={row['ROC_AUC']:.3f}"
            )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "ROC_AUC",
    ascending=False
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print("\nTOP MODEL RESULTS")
print(results_df.head(15).round(3).to_string(index=False))
print("\nSaved:", RESULT_FILE)