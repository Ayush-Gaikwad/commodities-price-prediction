
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

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FEATURES_PATH = "data/processed/sugar_features.csv"
PREDICTIONS_PATH = "results/predictions.csv"

RESULTS_DIR = "results"
GRAPHS_DIR = "results/graphs"

TEST_SIZE = 0.20

MODEL_COLUMNS = {
    "Linear Regression": "linear_regression_prediction",
    "Ridge Regression": "ridge_regression_prediction",
    "ElasticNet": "elasticnet_prediction",
    "Random Forest": "random_forest_prediction",
    "XGBoost": "xgboost_prediction"
}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    print("Loading evaluation datasets...")

    features = pd.read_csv(FEATURES_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    features["date"] = pd.to_datetime(features["date"])
    predictions["date"] = pd.to_datetime(predictions["date"])

    features = features.sort_values("date").reset_index(drop=True)
    predictions = predictions.sort_values("date").reset_index(drop=True)

    print("Feature dataset shape:", features.shape)
    print("Prediction dataset shape:", predictions.shape)

    return features, predictions


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    directional_accuracy = np.mean(
        np.sign(actual) == np.sign(predicted)
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy
    }


# --------------------------------------------------
# EVALUATE MODELS
# --------------------------------------------------

def evaluate_all_models(features, predictions):

    print("\nEvaluating all five models...")

    split_index = int(len(features) * (1 - TEST_SIZE))

    expected_test = features.iloc[split_index:].reset_index(drop=True)

    # Verify that prediction dates match the test dates
    if not predictions["date"].reset_index(drop=True).equals(
        expected_test["date"]
    ):
        raise ValueError(
            "Prediction dates do not match the chronological test split."
        )

    actual = predictions["actual"].to_numpy()

    # Verify actual target values
    expected_actual = expected_test["target"].to_numpy()

    if not np.allclose(actual, expected_actual, rtol=1e-8, atol=1e-8):
        raise ValueError(
            "Actual target values do not match the feature dataset."
        )

    results = {}

    # Zero-return baseline
    baseline_predictions = np.zeros_like(actual)

    baseline_metrics = calculate_metrics(
        actual,
        baseline_predictions
    )

    # Directional accuracy is not meaningful for a zero-return baseline
    baseline_metrics["Directional Accuracy"] = np.nan

    results["Zero-Return Baseline"] = baseline_metrics

    # Evaluate all five ML models
    for model_name, column in MODEL_COLUMNS.items():

        if column not in predictions.columns:
            raise ValueError(
                f"Missing prediction column: {column}"
            )

        results[model_name] = calculate_metrics(
            actual,
            predictions[column].to_numpy()
        )

    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "Model"
    metrics_df = metrics_df.reset_index()

    print("\nMODEL EVALUATION RESULTS")
    print("=" * 90)
    print(metrics_df.to_string(index=False))

    return metrics_df


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

def save_metrics(metrics_df):

    metrics_df.to_csv(
        f"{RESULTS_DIR}/model_comparison.csv",
        index=False
    )

    print("\nMetrics saved to results/model_comparison.csv")


# --------------------------------------------------
# GRAPH 1: ACTUAL VS PREDICTED RETURNS
# --------------------------------------------------

def plot_actual_vs_predicted(predictions):

    print("\nGenerating actual vs predicted graph...")

    plt.figure(figsize=(16, 7))

    plt.plot(
        predictions["date"],
        predictions["actual"],
        label="Actual Return",
        linewidth=2
    )

    for model_name, column in MODEL_COLUMNS.items():

        plt.plot(
            predictions["date"],
            predictions[column],
            label=model_name,
            alpha=0.75
        )

    plt.title("Actual vs Predicted Next-Day Sugar Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{GRAPHS_DIR}/08_actual_vs_predicted.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# GRAPH 2: METRIC COMPARISON
# --------------------------------------------------

def plot_metric_comparison(metrics_df):

    print("Generating metric comparison graphs...")

    metrics = [
        "MAE",
        "RMSE",
        "R2",
        "Directional Accuracy"
    ]

    for metric in metrics:

        plt.figure(figsize=(11, 6))

        sns.barplot(
            data=metrics_df,
            x="Model",
            y=metric
        )

        plt.title(f"Model Comparison - {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()

        filename = metric.lower().replace(" ", "_")

        plt.savefig(
            f"{GRAPHS_DIR}/09_{filename}_comparison.png",
            dpi=300
        )

        plt.close()


# --------------------------------------------------
# GRAPH 3: RESIDUAL DISTRIBUTIONS
# --------------------------------------------------

def plot_residuals(predictions):

    print("Generating residual distribution graphs...")

    for model_name, column in MODEL_COLUMNS.items():

        residuals = (
            predictions["actual"] - predictions[column]
        )

        plt.figure(figsize=(10, 6))

        sns.histplot(
            residuals,
            bins=40,
            kde=True
        )

        plt.axvline(x=0, linestyle="--")

        plt.title(f"Residual Distribution - {model_name}")
        plt.xlabel("Prediction Error (percentage points)")
        plt.ylabel("Frequency")
        plt.tight_layout()

        filename = model_name.lower().replace(" ", "_")

        plt.savefig(
            f"{GRAPHS_DIR}/10_residuals_{filename}.png",
            dpi=300
        )

        plt.close()


# --------------------------------------------------
# IDENTIFY BEST MODEL
# --------------------------------------------------

def identify_best_model(metrics_df):

    ml_results = metrics_df[
        metrics_df["Model"] != "Zero-Return Baseline"
    ]

    best_model = ml_results.loc[
        ml_results["RMSE"].idxmin()
    ]

    baseline = metrics_df[
        metrics_df["Model"] == "Zero-Return Baseline"
    ].iloc[0]

    print("\nBEST MACHINE LEARNING MODEL")
    print("=" * 60)

    print("Model:", best_model["Model"])
    print("MAE:", round(best_model["MAE"], 4))
    print("RMSE:", round(best_model["RMSE"], 4))
    print("R2:", round(best_model["R2"], 4))
    print(
        "Directional Accuracy:",
        round(best_model["Directional Accuracy"], 2),
        "%"
    )

    print("\nCOMPARISON WITH ZERO-RETURN BASELINE")

    if best_model["RMSE"] < baseline["RMSE"]:
        print("The best ML model has a lower RMSE than the baseline.")
    else:
        print("The best ML model does not beat the zero-return baseline.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("PHASE 6: MODEL EVALUATION")
    print("Five Regression Models - 11 Features")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    features, predictions = load_data()

    metrics_df = evaluate_all_models(
        features,
        predictions
    )

    save_metrics(metrics_df)

    plot_actual_vs_predicted(predictions)

    plot_metric_comparison(metrics_df)

    plot_residuals(predictions)

    identify_best_model(metrics_df)

    print("\n" + "=" * 65)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
