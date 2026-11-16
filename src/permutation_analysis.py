
# ============================================================
# PHASE 7: PERMUTATION FEATURE IMPORTANCE ANALYSIS
# Five Regression Models
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/sugar_features.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_REPEATS = 10

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Ridge Regression": "ridge_regression.pkl",
    "ElasticNet": "elasticnet.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}


# ============================================================
# LOAD DATA AND FEATURE NAMES
# ============================================================

def load_data():

    print("Loading dataset and feature names...")

    data = pd.read_csv(DATA_PATH)

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    with open(
        os.path.join(MODEL_DIR, "feature_columns.json"),
        "r"
    ) as file:

        feature_columns = json.load(file)

    missing = [
        column for column in feature_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    X = data[feature_columns].copy()

    y = data["target"].copy()

    split_index = int(
        len(data) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Dataset shape:", data.shape)
    print("Number of features:", len(feature_columns))
    print("Test observations:", len(X_test))

    return X_train, X_test, y_train, y_test, feature_columns


# ============================================================
# CALCULATE PERMUTATION IMPORTANCE
# ============================================================

def analyze_models(X_test, y_test, feature_columns):

    print("\nCalculating permutation importance...")

    all_results = []

    for model_name, filename in MODEL_FILES.items():

        print(f"\nAnalyzing {model_name}...")

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        model = joblib.load(model_path)

        result = permutation_importance(
            estimator=model,
            X=X_test,
            y=y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

        importance_df = pd.DataFrame({

            "Feature": feature_columns,

            "Importance Mean": result.importances_mean,

            "Importance Std": result.importances_std,

            "Model": model_name

        })

        importance_df = importance_df.sort_values(
            "Importance Mean",
            ascending=False
        ).reset_index(drop=True)

        print(f"\nTop 10 features - {model_name}")

        print(
            importance_df.head(10).to_string(index=False)
        )

        all_results.append(importance_df)

    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    return final_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(importance_df):

    output_path = (
        f"{RESULTS_DIR}/permutation_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nPermutation importance saved to {output_path}"
    )


# ============================================================
# GENERATE INDIVIDUAL MODEL GRAPHS
# ============================================================

def generate_graphs(importance_df):

    print("\nGenerating feature importance graphs...")

    graph_dir = f"{RESULTS_DIR}/permutation_graphs"

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    for model_name in MODEL_FILES:

        model_data = importance_df[
            importance_df["Model"] == model_name
        ].nlargest(
            10,
            "Importance Mean"
        )

        model_data = model_data.sort_values(
            "Importance Mean",
            ascending=True
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            model_data["Feature"],
            model_data["Importance Mean"],
            xerr=model_data["Importance Std"]
        )

        plt.axvline(
            x=0,
            linestyle="--"
        )

        plt.title(
            f"Permutation Feature Importance - {model_name}"
        )

        plt.xlabel(
            "Increase in RMSE After Shuffling"
        )

        plt.ylabel("Feature")

        plt.tight_layout()

        filename = (
            model_name.lower().replace(" ", "_")
            + "_permutation.png"
        )

        plt.savefig(
            os.path.join(graph_dir, filename),
            dpi=300
        )

        plt.close()


# ============================================================
# GENERATE COMBINED FEATURE COMPARISON
# ============================================================

def generate_combined_summary(importance_df):

    summary = importance_df.pivot(
        index="Feature",
        columns="Model",
        values="Importance Mean"
    )

    summary.to_csv(
        f"{RESULTS_DIR}/permutation_importance_summary.csv"
    )

    print(
        "\nCombined feature importance summary saved."
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)

    print("PERMUTATION FEATURE IMPORTANCE ANALYSIS")

    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns
    ) = load_data()

    importance_df = analyze_models(
        X_test,
        y_test,
        feature_columns
    )

    save_results(importance_df)

    generate_graphs(importance_df)

    generate_combined_summary(importance_df)

    print("\n" + "=" * 70)

    print("PERMUTATION ANALYSIS COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print("\nGenerated files:")

    print("- results/permutation_importance.csv")

    print("- results/permutation_importance_summary.csv")

    print("- results/permutation_graphs/")


if __name__ == "__main__":

    main()
