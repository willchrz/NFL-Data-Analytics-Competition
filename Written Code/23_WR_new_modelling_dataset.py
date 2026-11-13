# NFL BIG DATA BOWL 2027
# Another WR Modeling dataset... I want to cry
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(r"W:\nfl-big-data-bowl-2027")
OUTPUT = ROOT / "Output Data"

BASE_FILE = OUTPUT / "WR_Model_MovementOnly.csv"
DRILL_FILE = OUTPUT / "WR_Model_MovementPlusDrills.csv"
RESULT_FILE = OUTPUT / "WR_Final_Movement_Model_Results.csv"

print("\nSCRIPT 23: FINAL WR MODEL COMPARISON")

baseline = pd.read_csv(BASE_FILE)
drills = pd.read_csv(DRILL_FILE)

TARGET = "target"

print("\nBaseline shape:", baseline.shape)
print("Drill dataset shape:", drills.shape)

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
    if c in baseline.columns
]

print("\nTraditional combine features:")
print(traditional_features)

metadata = [
    "nfl_id",
    "display_name",
    "draft_year",
    TARGET
]

excluded_keywords = [
    "ngs",
    "college",
    "draft",
    "career",
    "receiving",
    "success",
    "label",
    "hit",
    "bust"
]

movement_features = [
    c for c in baseline.columns
    if c not in metadata + traditional_features
    and not any(
        keyword in c.lower()
        for keyword in excluded_keywords
    )
    and pd.api.types.is_numeric_dtype(baseline[c])
]

print("\nTracking movement features:")
print(movement_features)

combine_drills = [
    "FORTY_YARD_DASH",
    "GAUNTLET_DRILL",
    "SHORT_SHUTTLE",
    "THREE_CONE_DRILL"
]

route_drills = [
    "SLANT_ROUTE_LEFT",
    "DAGGER_ROUTE_LEFT",
    "POST_CORNER_ROUTE_RIGHT",
    "RED_ZONE_FADE_RIGHT",
    "CURL_ROUTE_RIGHT",
    "COMEBACK_ROUTE_RIGHT",
    "SLOT_SAIL_ROUTE_RIGHT",
    "GO_ROUTE_RIGHT",
    "SLOT_STRIKE_ROUTE_LEFT",
    "OVER_SHOULDER_ADJUST",
    "SPEED_OUT_ROUTE_LEFT"
]

def select_drill_features(df, drill_names):

    selected = []

    for col in df.columns:

        if col in metadata:
            continue

        if any(
            col.startswith(drill + "_")
            for drill in drill_names
        ):
            selected.append(col)

    # Keep features with reasonable coverage
    selected = [
        c for c in selected
        if df[c].isna().mean() <= 0.50
    ]

    # Remove constant features
    selected = [
        c for c in selected
        if df[c].nunique(dropna=True) > 1
    ]

    return selected

combine_drill_features = select_drill_features(
    drills, combine_drills
)

route_drill_features = select_drill_features(
    drills, route_drills
)

# Limit feature counts to reduce overfitting
combine_drill_features = combine_drill_features[:25]
route_drill_features = route_drill_features[:25]

drill_features = list(dict.fromkeys(
    combine_drill_features + route_drill_features
))[:40]

print("\nSelected combine drill features:")
print(combine_drill_features)

print("\nSelected route drill features:")
print(route_drill_features)

print("\nTotal selected drill features:", len(drill_features))

college_features = [
    "ngs_college_production_score"
]

college_features = [
    c for c in college_features
    if c in baseline.columns
]

print("\nNGS College Production Features:")
print(college_features)

traditional = traditional_features

tracking = movement_features

traditional_tracking = list(dict.fromkeys(
    traditional + tracking
))

traditional_tracking_drills = list(dict.fromkeys(
    traditional + tracking + drill_features
))

traditional_tracking_drills_college = list(dict.fromkeys(
    traditional + tracking + drill_features + college_features
))

feature_sets = {

    "Model 1 - Traditional Combine":
        traditional,

    "Model 2 - Tracking Movement":
        tracking,

    "Model 3 - Traditional + Tracking":
        traditional_tracking,

    "Model 4 - Traditional + Tracking + Drills":
        traditional_tracking_drills,

    "Model 5 - Traditional + Tracking + Drills + College":
        traditional_tracking_drills_college
}

y = baseline[TARGET].astype(int)

print("\nTarget distribution:")
print(y.value_counts())

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

results = []

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

for dataset_name, features in feature_sets.items():

    # Choose correct dataframe
    if any(
        feature in drills.columns
        for feature in features
    ) and any(
        feature in drill_features
        for feature in features
    ):
        X = drills[features]
    else:
        X = baseline[features]

    print("\n" + "=" * 60)
    print(dataset_name)
    print("Predictors:", len(features))
    print("=" * 60)

    for model_name, model in models.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        row = {
            "Feature_Group": dataset_name,
            "Model": model_name,
            "Predictor_Count": len(features),
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC_AUC": scores["test_roc_auc"].mean()
        }

        results.append(row)

        print(
            f"{model_name}: "
            f"Accuracy={row['Accuracy']:.3f}, "
            f"Precision={row['Precision']:.3f}, "
            f"Recall={row['Recall']:.3f}, "
            f"F1={row['F1']:.3f}, "
            f"AUC={row['ROC_AUC']:.3f}"
        )

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print("\nFINAL MODEL RESULTS")
print(results_df.round(3).to_string(index=False))
print("\nSaved:", RESULT_FILE)
