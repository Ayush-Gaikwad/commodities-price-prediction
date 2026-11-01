
# ============================================================
# PHASE 6: MODEL EVALUATION
# Commodities Price Prediction Using Machine Learning Regression
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = "data/processed/sugar_features.csv"

PREDICTIONS_PATH = "results/predictions.csv"

RESULTS_DIR = "results"

GRAPHS_DIR = "results/graphs"

TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading evaluation datasets...")

    features = pd.read_csv(
        FEATURES_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    features["date"] = pd.to_datetime(
        features["date"]
    )

    features["target_date"] = pd.to_datetime(
        features["target_date"]
    )

    predictions["target_date"] = pd.to_datetime(
        predictions["target_date"]
    )

    features = features.sort_values(
        "date"
    ).reset_index(drop=True)

    predictions = predictions.sort_values(
        "target_date"
    ).reset_index(drop=True)

    print(
        "Feature dataset shape:",
        features.shape
    )

    print(
        "Prediction dataset shape:",
        predictions.shape
    )

    return features, predictions


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100

    return {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "MAPE": mape

    }


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(
    features,
    predictions
):

    print("\nEvaluating models...")

    split_index = int(
        len(features) * (1 - TEST_SIZE)
    )

    # Persistence baseline:
    # Predict next-day price using today's closing price.
    baseline_predictions = (
        features["close"]
        .iloc[split_index:]
        .to_numpy()
    )

    actual = predictions[
        "actual"
    ].to_numpy()

    # Ensure prediction lengths match.
    if len(actual) != len(baseline_predictions):

        raise ValueError(
            "Prediction length does not match "
            "the chronological test split."
        )

    results = {}

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    results["Persistence Baseline"] = (
        calculate_metrics(
            actual,
            baseline_predictions
        )
    )

    # --------------------------------------------------------
    # LINEAR REGRESSION
    # --------------------------------------------------------

    results["Linear Regression"] = (
        calculate_metrics(

            actual,

            predictions[
                "linear_regression_prediction"
            ].to_numpy()

        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    results["Random Forest"] = (
        calculate_metrics(

            actual,

            predictions[
                "random_forest_prediction"
            ].to_numpy()

        )
    )

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    results["XGBoost"] = (
        calculate_metrics(

            actual,

            predictions[
                "xgboost_prediction"
            ].to_numpy()

        )
    )

    metrics_df = pd.DataFrame(
        results
    ).T

    metrics_df.index.name = "Model"

    metrics_df = metrics_df.reset_index()

    print("\nMODEL EVALUATION RESULTS")

    print("=" * 80)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    return metrics_df


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics_df):

    metrics_df.to_csv(

        f"{RESULTS_DIR}/model_comparison.csv",

        index=False

    )

    print(
        "\nMetrics saved to results/model_comparison.csv"
    )


# ============================================================
# GRAPH 1: ACTUAL VS PREDICTED
# ============================================================

def plot_actual_vs_predicted(
    predictions
):

    print(
        "\nGenerating actual vs predicted graph..."
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(

        predictions["target_date"],

        predictions["actual"],

        label="Actual Price",

        linewidth=2

    )

    plt.plot(

        predictions["target_date"],

        predictions[
            "linear_regression_prediction"
        ],

        label="Linear Regression",

        alpha=0.8

    )

    plt.plot(

        predictions["target_date"],

        predictions[
            "random_forest_prediction"
        ],

        label="Random Forest",

        alpha=0.8

    )

    plt.plot(

        predictions["target_date"],

        predictions[
            "xgboost_prediction"
        ],

        label="XGBoost",

        alpha=0.8

    )

    plt.title(
        "Actual vs Predicted Sugar Prices",
        fontsize=16
    )

    plt.xlabel("Date")

    plt.ylabel("Closing Price")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(

        f"{GRAPHS_DIR}/08_actual_vs_predicted.png",

        dpi=300

    )

    plt.close()


# ============================================================
# GRAPH 2: METRIC COMPARISON
# ============================================================

def plot_metric_comparison(
    metrics_df
):

    print(
        "Generating model comparison graph..."
    )

    metrics = [
        "MAE",
        "RMSE",
        "MAPE"
    ]

    for metric in metrics:

        plt.figure(
            figsize=(10, 6)
        )

        sns.barplot(

            data=metrics_df,

            x="Model",

            y=metric

        )

        plt.title(
            f"Model Comparison - {metric}",
            fontsize=16
        )

        plt.xlabel("Model")

        plt.ylabel(metric)

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        plt.savefig(

            f"{GRAPHS_DIR}/09_{metric.lower()}_comparison.png",

            dpi=300

        )

        plt.close()


# ============================================================
# GRAPH 3: RESIDUAL DISTRIBUTION
# ============================================================

def plot_residuals(
    predictions
):

    print(
        "Generating residual distribution graph..."
    )

    models = {

        "Linear Regression":
        "linear_regression_prediction",

        "Random Forest":
        "random_forest_prediction",

        "XGBoost":
        "xgboost_prediction"

    }

    for model_name, column in models.items():

        residuals = (

            predictions["actual"]

            - predictions[column]

        )

        plt.figure(
            figsize=(10, 6)
        )

        sns.histplot(

            residuals,

            bins=40,

            kde=True

        )

        plt.axvline(
            x=0,
            linestyle="--"
        )

        plt.title(
            f"Residual Distribution - {model_name}",
            fontsize=16
        )

        plt.xlabel(
            "Prediction Error"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.tight_layout()

        filename = (

            model_name.lower()
            .replace(" ", "_")

        )

        plt.savefig(

            f"{GRAPHS_DIR}/10_residuals_{filename}.png",

            dpi=300

        )

        plt.close()


# ============================================================
# FIND BEST MODEL
# ============================================================

def identify_best_model(
    metrics_df
):

    # Lowest RMSE is our primary selection criterion.
    best_model = metrics_df.loc[
        metrics_df["RMSE"].idxmin()
    ]

    print("\nBEST MODEL")

    print("=" * 60)

    print(
        "Model:",
        best_model["Model"]
    )

    print(
        "MAE:",
        round(
            best_model["MAE"],
            4
        )
    )

    print(
        "RMSE:",
        round(
            best_model["RMSE"],
            4
        )
    )

    print(
        "R2:",
        round(
            best_model["R2"],
            4
        )
    )

    print(
        "MAPE:",
        round(
            best_model["MAPE"],
            4
        )
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)

    print(
        "PHASE 6: MODEL EVALUATION"
    )

    print("=" * 60)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        GRAPHS_DIR,
        exist_ok=True
    )

    features, predictions = load_data()

    metrics_df = evaluate_all_models(

        features,

        predictions

    )

    save_metrics(
        metrics_df
    )

    plot_actual_vs_predicted(
        predictions
    )

    plot_metric_comparison(
        metrics_df
    )

    plot_residuals(
        predictions
    )

    identify_best_model(
        metrics_df
    )

    print("\n" + "=" * 60)

    print(
        "MODEL EVALUATION COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()
