
# ============================================================
# PHASE 5: MACHINE LEARNING MODEL DEVELOPMENT
# ============================================================

import os

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/sugar_features.csv"

MODEL_DIR = "models"

RESULTS_DIR = "results"

TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading engineered dataset...")

    data = pd.read_csv(DATA_PATH)

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data["target_date"] = pd.to_datetime(
        data["target_date"]
    )

    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    print("Dataset shape:", data.shape)

    return data


# ============================================================
# SELECT FEATURES
# ============================================================

def select_features(data):

    feature_columns = [

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
        "open_close_diff"

    ]

    # Include volume if it exists in the dataset.
    if "volume" in data.columns:

        feature_columns.append(
            "volume"
        )

    # Keep only features that exist.
    feature_columns = [

        column

        for column in feature_columns

        if column in data.columns

    ]

    X = data[
        feature_columns
    ].copy()

    y = data[
        "target"
    ].copy()

    print("\nSelected features:")

    print(feature_columns)

    print("\nNumber of features:")

    print(len(feature_columns))

    return X, y, feature_columns


# ============================================================
# CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

def split_data(X, y, data):

    print("\nPerforming chronological split...")

    split_index = int(
        len(data) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    train_dates = data[
        "date"
    ].iloc[:split_index]

    test_dates = data[
        "target_date"
    ].iloc[split_index:]

    print(
        f"Training observations: {len(X_train)}"
    )

    print(
        f"Testing observations: {len(X_test)}"
    )

    print(
        "\nTraining period:"
    )

    print(
        train_dates.min(),
        "to",
        train_dates.max()
    )

    print(
        "\nTesting target period:"
    )

    print(
        test_dates.min(),
        "to",
        test_dates.max()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        test_dates
    )


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    X_train,
    y_train,
    X_test
):

    print("\nTraining Linear Regression...")

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_predictions = linear_model.predict(
        X_test
    )

    print("Linear Regression completed.")

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    random_forest_model = RandomForestRegressor(

        n_estimators=300,

        max_depth=12,

        min_samples_split=5,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1

    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    random_forest_predictions = (
        random_forest_model.predict(
            X_test
        )
    )

    print("Random Forest completed.")

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    print("\nTraining XGBoost...")

    xgboost_model = XGBRegressor(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=4,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_lambda=1.0,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=42,

        n_jobs=-1

    )

    xgboost_model.fit(
        X_train,
        y_train
    )

    xgboost_predictions = (
        xgboost_model.predict(
            X_test
        )
    )

    print("XGBoost completed.")

    # --------------------------------------------------------
    # STORE MODELS AND PREDICTIONS
    # --------------------------------------------------------

    models = {

        "Linear Regression": linear_model,

        "Random Forest": random_forest_model,

        "XGBoost": xgboost_model

    }

    predictions = {

        "Linear Regression": (
            linear_predictions
        ),

        "Random Forest": (
            random_forest_predictions
        ),

        "XGBoost": (
            xgboost_predictions
        )

    }

    return models, predictions


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(models):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    for name, model in models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        joblib.dump(
            model,
            path
        )

        print(
            f"Saved {name} to {path}"
        )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    y_test,
    predictions,
    test_dates
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results = pd.DataFrame({

        "target_date": (
            test_dates.to_numpy()
        ),

        "actual": (
            y_test.to_numpy()
        )

    })

    for name, values in predictions.items():

        column_name = (

            name.lower()
            .replace(" ", "_")
            + "_prediction"

        )

        results[column_name] = values

    results.to_csv(

        f"{RESULTS_DIR}/predictions.csv",

        index=False

    )

    print(
        "\nPredictions saved to results/predictions.csv"
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)

    print(
        "PHASE 5: MACHINE LEARNING MODEL DEVELOPMENT"
    )

    print("=" * 60)

    data = load_data()

    X, y, feature_columns = (
        select_features(data)
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_dates
    ) = split_data(
        X,
        y,
        data
    )

    models, predictions = train_models(

        X_train,

        y_train,

        X_test

    )

    save_models(
        models
    )

    save_predictions(

        y_test,

        predictions,

        test_dates

    )

    print(
        "\nAll models trained successfully."
    )


if __name__ == "__main__":

    main()
