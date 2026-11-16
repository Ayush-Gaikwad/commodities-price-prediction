
# ============================================================
# MODEL DIAGNOSTICS
# Commodities Price Prediction
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = "data/processed/sugar_features.csv"
PREDICTIONS_PATH = "results/predictions.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_lag_5",
    "close_lag_10",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_50",
    "return_1",
    "return_5",
    "volatility_10",
    "volatility_20",
    "high_low_diff",
    "open_close_diff",
    "volume"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MODEL DIAGNOSTICS")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load datasets
    data = pd.read_csv(FEATURES_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    actual = predictions["actual"]

    # --------------------------------------------------------
    # 1. TARGET DISTRIBUTION
    # --------------------------------------------------------

    print("\n1. ACTUAL RETURN DISTRIBUTION")
    print("-" * 65)

    print(actual.describe())

    # --------------------------------------------------------
    # 2. PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    print("\n2. PREDICTION DISTRIBUTION")
    print("-" * 65)

    prediction_columns = {
        "Linear Regression": "linear_regression_prediction",
        "Random Forest": "random_forest_prediction",
        "XGBoost": "xgboost_prediction"
    }

    distribution = []

    for model_name, column in prediction_columns.items():

        predicted = predictions[column]

        print(f"\n{model_name}")

        print("Mean:", round(predicted.mean(), 4))
        print("Standard deviation:", round(predicted.std(), 4))
        print("Minimum:", round(predicted.min(), 4))
        print("Maximum:", round(predicted.max(), 4))

        distribution.append({
            "Model": model_name,
            "Mean": predicted.mean(),
            "Standard Deviation": predicted.std(),
            "Minimum": predicted.min(),
            "Maximum": predicted.max()
        })

    distribution_df = pd.DataFrame(distribution)

    distribution_df.to_csv(
        f"{RESULTS_DIR}/prediction_distribution.csv",
        index=False
    )

    # --------------------------------------------------------
    # 3. FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\n\n3. FEATURE IMPORTANCE")
    print("-" * 65)

    importance_results = []

    model_files = {
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    for model_name, filename in model_files.items():

        model = joblib.load(
            os.path.join(MODEL_DIR, filename)
        )

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": importances,
            "Model": model_name
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        print(f"\n{model_name} - Top 10 Features")

        print(
            importance_df.head(10).to_string(index=False)
        )

        importance_results.append(importance_df)

    final_importance = pd.concat(
        importance_results,
        ignore_index=True
    )

    final_importance.to_csv(
        f"{RESULTS_DIR}/feature_importance.csv",
        index=False
    )

    # --------------------------------------------------------
    # 4. FEATURE IMPORTANCE GRAPH
    # --------------------------------------------------------

    for model_name in ["Random Forest", "XGBoost"]:

        model_data = final_importance[
            final_importance["Model"] == model_name
        ].nlargest(10, "Importance")

        plt.figure(figsize=(10, 6))

        plt.barh(
            model_data["Feature"],
            model_data["Importance"]
        )

        plt.gca().invert_yaxis()

        plt.title(
            f"Top 10 Feature Importance - {model_name}"
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")

        plt.tight_layout()

        filename = model_name.lower().replace(" ", "_")

        plt.savefig(
            f"{RESULTS_DIR}/{filename}_feature_importance.png",
            dpi=300
        )

        plt.close()

    # --------------------------------------------------------
    # 5. ACTUAL VS PREDICTED DISTRIBUTION GRAPH
    # --------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.hist(
        actual,
        bins=40,
        alpha=0.6,
        label="Actual Returns"
    )

    for model_name, column in prediction_columns.items():

        plt.hist(
            predictions[column],
            bins=40,
            alpha=0.4,
            label=model_name
        )

    plt.title("Actual vs Predicted Return Distribution")
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Frequency")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/return_distribution_comparison.png",
        dpi=300
    )

    plt.close()

    print("\n" + "=" * 65)
    print("MODEL DIAGNOSTICS COMPLETED")
    print("=" * 65)

    print("\nFiles generated:")
    print("- results/prediction_distribution.csv")
    print("- results/feature_importance.csv")
    print("- results/random_forest_feature_importance.png")
    print("- results/xgboost_feature_importance.png")
    print("- results/return_distribution_comparison.png")


if __name__ == "__main__":
    main()
