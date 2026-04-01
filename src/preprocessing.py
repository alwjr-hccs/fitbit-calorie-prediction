"""
preprocessing.py — Data cleaning, imputation, and feature engineering.

Functions
---------
load_and_clean(path)
    Load CSV, drop duplicates, handle missing values with median imputation.
engineer_features(df)
    Add step_intensity and activity_ratio columns.
prepare_pipeline(df, target_col)
    Full preprocessing: clean → engineer → split → scale.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean(path: str) -> pd.DataFrame:
    """Load a CSV file and perform basic cleaning.

    Parameters
    ----------
    path : str
        File path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with duplicates removed and missing
        numeric values filled using column medians.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates()

    # Median imputation for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {col}: {df[col].isnull().sum()} remaining NaNs "
                  f"(median={median_val:.2f})")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for the calorie prediction model.

    New columns
    -----------
    step_intensity : float
        Steps per active minute (total_steps / total_active_minutes).
    activity_ratio : float
        Fraction of tracked time spent in any active category.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain activity-minute and step columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with two new engineered columns appended.
    """
    df = df.copy()

    # Total active minutes
    active_cols = [c for c in df.columns
                   if "active" in c.lower() and "sedentary" not in c.lower()
                   and "minute" in c.lower()]
    if active_cols:
        df["total_active_minutes"] = df[active_cols].sum(axis=1)
    else:
        df["total_active_minutes"] = 0

    # step_intensity — steps per active minute (avoid division by zero)
    if "TotalSteps" in df.columns:
        step_col = "TotalSteps"
    elif "total_steps" in df.columns:
        step_col = "total_steps"
    else:
        step_col = None

    if step_col:
        df["step_intensity"] = np.where(
            df["total_active_minutes"] > 0,
            df[step_col] / df["total_active_minutes"],
            0.0,
        )
    else:
        df["step_intensity"] = 0.0

    # activity_ratio — active minutes as fraction of total tracked time
    sedentary_col = [c for c in df.columns if "sedentary" in c.lower()
                     and "minute" in c.lower()]
    if sedentary_col:
        total_time = df["total_active_minutes"] + df[sedentary_col[0]]
        df["activity_ratio"] = np.where(
            total_time > 0,
            df["total_active_minutes"] / total_time,
            0.0,
        )
    else:
        df["activity_ratio"] = 0.0

    return df


def prepare_pipeline(
    df: pd.DataFrame,
    target_col: str = "Calories",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """End-to-end preprocessing: clean → engineer → split → scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially cleaned dataframe.
    target_col : str
        Name of the target column.
    test_size : float
        Proportion of data reserved for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    df = engineer_features(df)

    # Select numeric features, drop target and ID-like columns
    drop_cols = [target_col]
    id_cols = [c for c in df.columns if "id" in c.lower()]
    drop_cols.extend(id_cols)

    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in drop_cols
    ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
