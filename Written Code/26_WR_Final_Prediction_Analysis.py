# NFL BIG DATA BOWL 2027
# WR Last Model for now... Gradient Boosting will return in Avengers Endgame
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
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

PREDICTION_FILE = OUTPUT / "WR_Final_OutOfFold_Predictions.csv"
SUMMARY_FILE = OUTPUT / "WR_Final_Prediction_Summary.csv"

movement = pd.read_csv(MOVEMENT_FILE)
relative = pd.read_csv(RELATIVE_FILE)

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

features = (
    traditional_features
    + movement_features
    + college_features
)

X = df[features]

# MODELS

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

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# OUT-OF-FOLD PREDICTIONS

all_predictions = []
model_summaries = []

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(model_name)
    print("=" * 65)

    oof_probability = np.zeros(len(df))
    oof_prediction = np.zeros(len(df), dtype=int)

    selected_feature_counts = {}

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Impute using training fold only
        imputer = SimpleImputer(strategy="median")

        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Select features using training fold only
        selector = SelectKBest(
            score_func=f_classif,
            k=15
        )

        X_train_sel = selector.fit_transform(
            X_train_imp,
            y_train
        )

        X_test_sel = selector.transform(X_test_imp)

        selected_features = np.array(features)[
            selector.get_support()
        ]

        for feature in selected_features:
            selected_feature_counts[feature] = (
                selected_feature_counts.get(feature, 0) + 1
            )

        # Train and predict
        model.fit(X_train_sel, y_train)

        probabilities = model.predict_proba(
            X_test_sel
        )[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        oof_probability[test_idx] = probabilities
        oof_prediction[test_idx] = predictions

        print(
            f"Fold {fold} | "
            f"Test Players={len(test_idx)} | "
            f"Accuracy={accuracy_score(y_test, predictions):.3f}"
        )

    # MODEL METRICS

    cm = confusion_matrix(
        y,
        oof_prediction,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y, oof_prediction)
    precision = precision_score(y, oof_prediction, zero_division=0)
    recall = recall_score(y, oof_prediction, zero_division=0)
    f1 = f1_score(y, oof_prediction, zero_division=0)
    auc = roc_auc_score(y, oof_probability)

    model_summaries.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })

    print("\nConfusion Matrix")
    print("Rows = Actual | Columns = Predicted")
    print("             Bust   Hit")
    print(f"Bust         {tn:4d}   {fp:4d}")
    print(f"Hit          {fn:4d}   {tp:4d}")

    print("\nOut-of-Fold Metrics")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"AUC:       {auc:.3f}")

    prediction_df = df[
        ["nfl_id", "display_name", "draft_year", TARGET]
    ].copy()

    prediction_df["Model"] = model_name
    prediction_df["Predicted_Probability_Hit"] = oof_probability
    prediction_df["Predicted_Class"] = oof_prediction

    prediction_df["Prediction_Result"] = np.where(
        prediction_df[TARGET]
        == prediction_df["Predicted_Class"],
        "Correct",
        "Incorrect"
    )

    prediction_df["Prediction_Type"] = np.select(
        [
            (prediction_df[TARGET] == 1)
            & (prediction_df["Predicted_Class"] == 1),

            (prediction_df[TARGET] == 0)
            & (prediction_df["Predicted_Class"] == 0),

            (prediction_df[TARGET] == 0)
            & (prediction_df["Predicted_Class"] == 1),

            (prediction_df[TARGET] == 1)
            & (prediction_df["Predicted_Class"] == 0)
        ],
        [
            "True Hit",
            "True Bust",
            "False Hit",
            "False Bust"
        ],
        default="Unknown"
    )

    all_predictions.append(prediction_df)

    # Feature selection frequency
    feature_frequency = pd.DataFrame({
        "Feature": list(selected_feature_counts.keys()),
        "Selection_Count": list(selected_feature_counts.values())
    })

    feature_frequency["Model"] = model_name
    feature_frequency["Selection_Percentage"] = (
        feature_frequency["Selection_Count"] / 5
    )

    print("\nMost Frequently Selected Features")
    print(
        feature_frequency.sort_values(
            "Selection_Count",
            ascending=False
        ).head(15).to_string(index=False)
    )

final_predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

final_predictions.to_csv(
    PREDICTION_FILE,
    index=False
)

summary_df = pd.DataFrame(model_summaries)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(summary_df.round(3).to_string(index=False))

print("\nSaved:", PREDICTION_FILE)
print("Saved:", SUMMARY_FILE)