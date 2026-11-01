
# ============================================================
# PHASE 4: FEATURE ENGINEERING
# Commodities Price Prediction Using Machine Learning Regression
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/sugar_prices.csv"

OUTPUT_PATH = "data/processed/sugar_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading commodity price dataset...")

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values("date")

    data = data.drop_duplicates(
        subset=["date"]
    )

    data = data.reset_index(drop=True)

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    print("Dataset loaded:", data.shape)

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    print("\nCreating lag features...")

    data = data.copy()

    # --------------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------------

    for lag in [1, 2, 3, 5, 10]:

        data[f"close_lag_{lag}"] = (
            data["close"].shift(lag)
        )

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    for window in [5, 10, 20, 50]:

        data[f"ma_{window}"] = (
            data["close"]
            .rolling(window=window)
            .mean()
        )

    # --------------------------------------------------------
    # DAILY RETURNS
    # --------------------------------------------------------

    data["return_1"] = (
        data["close"].pct_change()
    )

    data["return_5"] = (
        data["close"].pct_change(periods=5)
    )

    # --------------------------------------------------------
    # ROLLING VOLATILITY
    # --------------------------------------------------------

    data["volatility_10"] = (
        data["close"]
        .pct_change()
        .rolling(window=10)
        .std()
    )

    data["volatility_20"] = (
        data["close"]
        .pct_change()
        .rolling(window=20)
        .std()
    )

    # --------------------------------------------------------
    # PRICE DIFFERENCES
    # --------------------------------------------------------

    data["high_low_diff"] = (
        data["high"] - data["low"]
    )

    data["open_close_diff"] = (
        data["close"] - data["open"]
    )

    # --------------------------------------------------------
    # TARGET VARIABLE
    # --------------------------------------------------------

    # Predict the next trading day's closing price.
    data["target"] = (
        data["close"].shift(-1)
    )

    # Store the date corresponding to the target.
    data["target_date"] = (
        data["date"].shift(-1)
    )

    return data


# ============================================================
# CLEAN FEATURE DATASET
# ============================================================

def clean_feature_data(data):

    print("\nCleaning feature dataset...")

    data = data.copy()

    # Remove rows containing missing feature values.
    # Initial rows are missing due to lag and rolling windows.
    # Final row is missing because the next-day target is unknown.

    data = data.dropna()

    # Remove infinite values.
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    data = data.reset_index(drop=True)

    return data


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)

    print("PHASE 4: FEATURE ENGINEERING")

    print("=" * 60)

    data = load_data()

    data = create_features(data)

    data = clean_feature_data(data)

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature engineering completed.")

    print("\nFinal dataset shape:")

    print(data.shape)

    print("\nFeature columns:")

    print(data.columns.tolist())

    print("\nFirst 5 rows:")

    print(data.head())

    print("\nMissing values:")

    print(data.isnull().sum())

    print(
        f"\nDataset saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()
