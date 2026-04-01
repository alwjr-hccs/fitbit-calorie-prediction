# Fitbit Calorie Prediction 🔥

**Predicting Daily Calorie Expenditure from Wearable Sensor Data Using Supervised Regression**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alwjr-hccs/fitbit-calorie-prediction/blob/main/notebooks/fitbit_calorie_prediction.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project builds and compares supervised regression models to predict **daily calorie expenditure** using wearable fitness tracker data. It demonstrates a complete ML pipeline — from data loading and exploratory analysis through feature engineering, model training, evaluation, and interpretation.

## Key Features

- **Three regression models** compared head-to-head: Linear Regression, Random Forest, Gradient Boosting
- **Feature engineering**: `step_intensity` (steps per active minute) and `activity_ratio` (active minutes as fraction of day)
- **Comprehensive evaluation**: MAE, RMSE, R² with visualizations
- **Reusable preprocessing pipeline** with median imputation and standardized scaling
- **Interpretability**: Feature importance analysis and residual diagnostics

## Dataset

This project uses the [Fitbit Fitness Tracker Dataset](https://www.kaggle.com/datasets/arashnic/fitbit) (Kaggle), which contains personal fitness tracker data from thirty Fitbit users including minute-level physical activity, heart rate, and sleep monitoring.

**Input features include:**

| Category | Features |
|---|---|
| Demographics | `age`, `gender`, `weight_kg`, `height_cm`, `bmi` |
| Activity | `total_steps`, `total_distance`, `very_active_minutes`, `fairly_active_minutes`, `lightly_active_minutes`, `sedentary_minutes` |
| Heart Rate | `resting_hr`, `avg_hr` |
| Sleep | `total_minutes_asleep`, `total_time_in_bed` |
| Engineered | `step_intensity`, `activity_ratio` |

**Target variable:** `calories` (daily calorie expenditure)

## Project Structure

```
fitbit-calorie-prediction/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── notebooks/
│   └── fitbit_calorie_prediction.ipynb   # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Data cleaning & feature engineering
│   └── evaluation.py          # Model evaluation utilities
├── data/
│   └── README.md              # Dataset download instructions
└── figures/
    └── README.md              # Generated plots stored here
```

## Quick Start

### Option 1: Google Colab (Recommended)

Click the **Open in Colab** badge above to run the notebook directly in your browser — no local setup required.

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/alwjr-hccs/fitbit-calorie-prediction.git
cd fitbit-calorie-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/fitbit_calorie_prediction.ipynb
```

## Results Summary

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | Baseline | Baseline | Baseline |
| Random Forest | Improved | Improved | Improved |
| Gradient Boosting | **Best** | **Best** | **Best** |

> Exact metrics are computed at runtime — see the notebook for full results.

## Models & Methodology

1. **Data Preprocessing**: Missing value imputation (median strategy), outlier detection, feature scaling (StandardScaler)
2. **Feature Engineering**: Two derived features — `step_intensity` and `activity_ratio` — capture normalized activity patterns
3. **Train/Test Split**: 80/20 stratified split with random seed for reproducibility
4. **Model Training**: Three regressors trained with scikit-learn defaults as baselines, then evaluated
5. **Evaluation**: MAE, RMSE, R² metrics plus residual plots, predicted-vs-actual scatter, and feature importance charts

## Course Context

Developed for **ITAI 1371 — Fundamentals of Machine Learning** midterm project at Houston Community College.

## Author

**Andrew Williams**

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
