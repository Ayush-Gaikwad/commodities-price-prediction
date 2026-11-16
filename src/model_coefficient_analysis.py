
# ============================================================
# MODEL COEFFICIENT AND PREDICTION ANALYSIS
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"
RESULTS_DIR = "results"

PREDICTIONS_PATH = "results/predictions.csv"
FEATURES_PATH = "data/processed/sugar_features.csv"

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Ridge Regression": "ridge_regression.pkl",
    "ElasticNet": "elasticnet.pkl"
}


# ============================================================
# EXTRACT ACTUAL ESTIMATOR
# ============================================================

def get_estimator(model):

    # Handle GridSearchCV models
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_

    # Handle Pipeline models
    if isinstance(model, Pipeline):
        model = model.named_steps["model"]

    return model


# ============================================================
# EXTRACT COEFFICIENTS
# ============================================================

def get_coefficients(model):

    estimator = get_estimator(model)

    if not hasattr(estimator, "coef_"):
        return None, None

    coefficients = np.asarray(
        estimator.coef_
    ).flatten()

    intercept = np.asarray(
        estimator.intercept_
    ).flatten()[0]

    return coefficients, intercept


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MODEL COEFFICIENT ANALYSIS")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load feature names
    with open(
        f"{MODEL_DIR}/feature_columns.json",
        "r"
    ) as file:
        feature_columns = json.load(file)

    # Load predictions
    predictions = pd.read_csv(PREDICTIONS_PATH)

    # --------------------------------------------------------
    # COEFFICIENT ANALYSIS
    # --------------------------------------------------------

    coefficient_results = []

    for model_name, filename in MODEL_FILES.items():

        print(f"\nAnalyzing {model_name}...")

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        model = joblib.load(model_path)

        coefficients, intercept = get_coefficients(model)

        if coefficients is None:
            print("Coefficients unavailable.")
            continue

        if len(coefficients) != len(feature_columns):
            print("ERROR: Feature and coefficient counts do not match.")
            continue

        for feature, coefficient in zip(
            feature_columns,
            coefficients
        ):

            coefficient_results.append({
                "Model": model_name,
                "Feature": feature,
                "Coefficient": coefficient,
                "Absolute Coefficient": abs(coefficient)
            })

        print("Intercept:", intercept)

        print(
            "Non-zero coefficients:",
            np.count_nonzero(coefficients)
        )

        print(
            "Total coefficients:",
            len(coefficients)
        )

        if isinstance(model, GridSearchCV):
            print("Best parameters:", model.best_params_)

    # Save coefficients
    coefficient_df = pd.DataFrame(
        coefficient_results
    )

    coefficient_df.to_csv(
        f"{RESULTS_DIR}/model_coefficients.csv",
        index=False
    )

    print("\nCoefficient results saved.")

    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    print("\nPREDICTION VARIABILITY")
    print("-" * 65)

    prediction_columns = [
        column for column in predictions.columns
        if column.endswith("_prediction")
    ]

    distribution_results = []

    for column in prediction_columns:

        values = predictions[column]

        print(f"\n{column}")

        print("Mean:", round(values.mean(), 6))
        print("Standard deviation:", round(values.std(), 6))
        print("Minimum:", round(values.min(), 6))
        print("Maximum:", round(values.max(), 6))

        distribution_results.append({
            "Model": column,
            "Mean": values.mean(),
            "Standard Deviation": values.std(),
            "Minimum": values.min(),
            "Maximum": values.max()
        })

    pd.DataFrame(
        distribution_results
    ).to_csv(
        f"{RESULTS_DIR}/model_prediction_variability.csv",
        index=False
    )

    # --------------------------------------------------------
    # FEATURE CORRELATION
    # --------------------------------------------------------

    data = pd.read_csv(FEATURES_PATH)

    correlation = data[
        feature_columns
    ].corr()

    correlation.to_csv(
        f"{RESULTS_DIR}/feature_correlation.csv"
    )

    print("\nFeature correlation matrix saved.")

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("COEFFICIENT ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nGenerated files:")
    print("- results/model_coefficients.csv")
    print("- results/model_prediction_variability.csv")
    print("- results/feature_correlation.csv")


if __name__ == "__main__":
    main()
