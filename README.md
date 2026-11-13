

# NFL Big Data Bowl 2027: Player Movement & Performance Analytics

## Project Overview

This project investigates how NFL player movement characteristics, athletic performance, and tracking data can be used to better understand and predict player performance.

The broader objective is to develop an analytical framework capable of identifying measurable characteristics associated with successful and unsuccessful NFL players across different positions.

The initial phase focuses on rookie wide receivers, using NFL combine movement and tracking data to engineer player-level movement profiles and investigate relationships between athletic characteristics and professional production.

Wide receivers represent the starting point of the project rather than its ultimate scope. Future development will expand the framework to additional positions and incorporate position-specific movement and performance characteristics.

## Research Objectives

- Identify movement characteristics associated with NFL player performance.
- Develop position-specific athletic and movement profiles.
- Engineer meaningful features from NFL tracking and combine data.
- Investigate relationships between athletic characteristics and professional production.
- Develop predictive models to evaluate player performance and success.
- Expand the analytical framework across NFL positions.


## Data Sources

The project utilizes NFL tracking and combine movement data.

### Tracking Data
- Player movement observations recorded at 10 Hz.
- Speed, acceleration, distance, direction, and position.
- Movement sequences collected during combine drills.

### Combine Data
- Forty-yard dash.
- Three-cone drill.
- Short shuttle.
- Wide receiver skill drills.

Raw datasets are excluded from this repository because of their size and data management requirements.

## Methodology

### 1. Data Exploration
- Examine dataset structure and coverage.
- Identify unique players, drills, and observations.
- Investigate missing values and data quality.

### 2. Feature Engineering
Develop movement features including:

- Average and maximum speed.
- Average and maximum acceleration.
- Acceleration variability.
- Distance traveled across movement sequences.
- Drill-specific movement characteristics.
- Aggregated player movement profiles.

### 3. Feature Integration
- Combine engineered movement characteristics.
- Integrate player-level information and performance outcomes.
- Develop modeling datasets.

### 4. Predictive Modeling
Modeling approaches explored include:

- Gradient Boosting.
- XGBoost.
- Feature importance analysis.
- Model optimization and evaluation.

Predictive modeling remains an ongoing component of the project.

## Repository Structure

```text
NFL-Data-Analytics-Competition/
│
├── Written Code/
│   ├── 01_WR_Data_Exploration.py
│   ├── 02_WR_Combine_Tracking_Exploration.py
│   ├── ...
│   └── 26_WR_Final_Prediction_Analysis.py
│
├── Revised Code/
│   ├── 01_WR_Model_Data.py
│   └── 02_WR_Final_Models.py
│
├── GitHub Script Upload/
│
├── .gitignore
└── README.md
```

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Git / GitHub

## Project Status

**Status: In Progress**

Completed work includes:
- Tracking data exploration.
- Combine movement feature engineering.
- Player movement profile development.
- Drill-specific feature engineering.
- Movement data quality investigations.
- Feature integration and modeling dataset development.

Current work focuses on refining predictive modeling, evaluating engineered features, and investigating relationships between movement characteristics and NFL performance.

## Author

**William Chrz**

Master of Business Analytics and Data Science  
Oklahoma State University

GitHub: [@willchrz](https://github.com/willchrz)
