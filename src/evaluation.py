"""
evaluation.py — Model evaluation utilities and visualization helpers.

Functions
---------
evaluate_model(model, X_test, y_test)
    Compute MAE, RMSE, R² for a fitted model.
compare_models(results_dict)
    Print a formatted comparison table.
plot_predictions(y_test, y_pred, title)
    Predicted-vs-actual scatter plot.
plot_residuals(y_test, y_pred, title)
    Residual distribution histogram.
plot_feature_importance(model, feature_names, title)
    Horizontal bar chart of feature importances.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """Return a dict of regression metrics for a fitted model.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : array-like
    y_test : array-like

    Returns
    -------
    dict with keys 'MAE', 'RMSE', 'R2' and array 'y_pred'.
    """
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "y_pred": y_pred,
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame from a dict of model results.

    Parameters
    ----------
    results : dict
        {model_name: metrics_dict} as returned by evaluate_model.

    Returns
    -------
    pd.DataFrame sorted by R² descending.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "MAE": round(metrics["MAE"], 2),
            "RMSE": round(metrics["RMSE"], 2),
            "R²": round(metrics["R2"], 4),
        })
    df = pd.DataFrame(rows).sort_values("R²", ascending=False)
    return df.reset_index(drop=True)


def plot_predictions(y_test, y_pred, title="Predicted vs Actual"):
    """Scatter plot of predicted vs actual values with identity line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3)
    lims = [
        min(min(y_test), min(y_pred)),
        max(max(y_test), max(y_pred)),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Calories")
    ax.set_ylabel("Predicted Calories")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_test, y_pred, title="Residual Distribution"):
    """Histogram of residuals (actual - predicted)."""
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, title="Feature Importance",
                            top_n=15):
    """Horizontal bar chart of feature importances (tree-based models)."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="teal",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    return fig
