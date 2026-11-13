# NFL BIG DATA BOWL 2027
# This is probably not the final model but I'm calling it that to give good vibes
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

MOVEMENT_FILE = OUTPUT / "WR_Model_MovementOnly.csv"
RELATIVE_FILE = OUTPUT / "WR_Model_Relative.csv"
RESULT_FILE = OUTPUT / "WR_Final_Model_Comparison.csv"
FEATURE_FILE = OUTPUT / "WR_Final_Selected_Features.csv"

movement = pd.read_csv(MOVEMENT_FILE)
relative = pd.read_csv(RELATIVE_FILE)

# College production score
college = relative[
    ["nfl_id", "ngs_college_production_score"]
].drop_duplicates(subset="nfl_id")

df = movement.merge(
    college,
    on="nfl_id",
    how="left"
)

TARGET = "target"
y = df[TARGET].astype(int)

metadata = ["nfl_id", "display_name", "draft_year", TARGET]

traditional_features = [
    c for c in [
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
    ] if c in df.columns
]

college_features = ["ngs_college_production_score"]

movement_features = [
    c for c in df.columns
    if c not in metadata + traditional_features + college_features
    and pd.api.types.is_numeric_dtype(df[c])
]

print("=" * 65)
print("Players:", len(df))
print("Hits:", int(y.sum()))
print("Busts:", int((y == 0).sum()))
print("Traditional features:", len(traditional_features))
print("Movement features:", len(movement_features))
print("College production features:", len(college_features))

feature_groups = {
    "Tracking Movement": movement_features,

    "Traditional + Tracking":
        traditional_features + movement_features,

    "Traditional + Tracking + College":
        traditional_features + movement_features + college_features
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
feature_selection_results = []

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
    )
}

for group_name, features in feature_groups.items():

    print("\n" + "=" * 65)
    print(group_name)
    print("Predictors:", len(features))
    print("=" * 65)

    X = df[features]

    # Select 15 features when possible
    k = min(15, len(features))

    for model_name, model in models.items():

        fold_accuracy = []
        fold_precision = []
        fold_recall = []
        fold_f1 = []
        fold_auc = []

        selected_counts = {}

        for train_idx, test_idx in cv.split(X, y):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Impute missing values using training data
            imputer = SimpleImputer(strategy="median")

            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            # Feature selection inside each training fold
            selector = SelectKBest(
                score_func=f_classif,
                k=k
            )

            X_train_sel = selector.fit_transform(
                X_train_imp,
                y_train
            )

            X_test_sel = selector.transform(X_test_imp)

            selected_columns = np.array(features)[
                selector.get_support()
            ]

            for feature in selected_columns:
                selected_counts[feature] = (
                    selected_counts.get(feature, 0) + 1
                )

            # Fit model
            model.fit(X_train_sel, y_train)

            predictions = model.predict(X_test_sel)
            probabilities = model.predict_proba(X_test_sel)[:, 1]

            fold_accuracy.append(
                accuracy_score(y_test, predictions)
            )

            fold_precision.append(
                precision_score(
                    y_test, predictions,
                    zero_division=0
                )
            )

            fold_recall.append(
                recall_score(
                    y_test, predictions,
                    zero_division=0
                )
            )

            fold_f1.append(
                f1_score(
                    y_test, predictions,
                    zero_division=0
                )
            )

            fold_auc.append(
                roc_auc_score(y_test, probabilities)
            )

        # Save averaged results
        results.append({
            "Feature_Group": group_name,
            "Model": model_name,
            "Selected_Features": k,
            "Accuracy": np.mean(fold_accuracy),
            "Precision": np.mean(fold_precision),
            "Recall": np.mean(fold_recall),
            "F1": np.mean(fold_f1),
            "ROC_AUC": np.mean(fold_auc)
        })

        # Save feature selection frequency
        for feature, count in selected_counts.items():

            feature_selection_results.append({
                "Feature_Group": group_name,
                "Model": model_name,
                "Feature": feature,
                "Selection_Count": count,
                "Selection_Percentage": count / cv.n_splits
            })

        print(
            f"{model_name} | "
            f"Accuracy={np.mean(fold_accuracy):.3f} | "
            f"F1={np.mean(fold_f1):.3f} | "
            f"AUC={np.mean(fold_auc):.3f}"
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

features_df = pd.DataFrame(feature_selection_results)

features_df = features_df.sort_values(
    ["Feature_Group", "Model", "Selection_Count"],
    ascending=[True, True, False]
)

features_df.to_csv(
    FEATURE_FILE,
    index=False
)

print("\n" + "=" * 65)
print("FINAL MODEL RESULTS")
print("=" * 65)

print(
    results_df.round(3).to_string(index=False)
)

print("\nMOST FREQUENTLY SELECTED FEATURES")
print("=" * 65)

print(
    features_df[
        features_df["Feature_Group"]
        == "Traditional + Tracking + College"
    ]
    .sort_values(
        "Selection_Count",
        ascending=False
    )
    .head(20)
    .round(3)
    .to_string(index=False)
)

print("\nSaved:", RESULT_FILE)
print("Saved:", FEATURE_FILE)