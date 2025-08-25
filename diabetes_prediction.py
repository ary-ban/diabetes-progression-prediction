"""
Diabetes Progression Prediction
-------------------------------

This script builds a regression model to predict disease progression in patients using the Diabetes dataset available in sklearn.datasets.

The dataset contains 10 baseline variables such as age, sex, BMI, blood pressure, and six blood serum measurements. The target variable measures the quantitative progression of diabetes one year after baseline.

Workflow:
1. Load the dataset and standardize features.
2. Split into training and test sets.
3. Train a RandomForestRegressor and LinearRegression model.
4. Evaluate models with R^2 and RMSE metrics and report the mean target value for reference.
"""

from __future__ import annotations
import math
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series]:
    """Load and standardise the diabetes progression dataset.

    Returns
    -------
    X : pd.DataFrame
        Standardised feature matrix.
    y : pd.Series
        Target variable representing a quantitative measure of disease progression.
    """
    data = load_diabetes(as_frame=True)
    X_raw: pd.DataFrame = data.data
    y: pd.Series = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = pd.DataFrame(X_scaled, columns=X_raw.columns)
    return X, y


def train_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train multiple regression models and evaluate their performance.

    Returns a dictionary containing metrics for Random Forest and Linear Regression models,
    including test R^2 and RMSE.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["random_forest"] = {
        "model": rf,
        "r2": r2_score(y_test, y_pred_rf),
        "rmse": math.sqrt(mean_squared_error(y_test, y_pred_rf)),
    }
    # Linear Regression baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["linear_regression"] = {
        "model": lr,
        "r2": r2_score(y_test, y_pred_lr),
        "rmse": math.sqrt(mean_squared_error(y_test, y_pred_lr)),
    }
    # Baseline mean value
    results["baseline_mean"] = float(y.mean())
    return results


def main() -> None:
    X, y = load_and_preprocess()
    results = train_models(X, y)
    print("Diabetes Progression Prediction Results")
    print("--------------------------------------")
    for name, res in results.items():
        if name == "baseline_mean":
            continue
        print(f"\nModel: {name.replace('_', ' ').title()}")
        print(f"R^2 Score: {res['r2']:.4f}")
        print(f"RMSE: {res['rmse']:.4f}")
    print(f"\nMean target value (baseline): {results['baseline_mean']:.4f}")


if __name__ == "__main__":
    main()
