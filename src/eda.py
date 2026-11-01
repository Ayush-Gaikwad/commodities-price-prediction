
# ============================================================
# COMMODITIES PRICE PREDICTION
# PHASE 3: EXPLORATORY DATA ANALYSIS
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/sugar_prices.csv"

OUTPUT_DIR = "results/graphs"

SUMMARY_PATH = "results/eda_summary.txt"

EDA_DATA_PATH = "data/processed/sugar_eda_data.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run src/data_collection.py first."
        )

    data = pd.read_csv(DATA_PATH)

    # Convert date column to datetime.
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # Remove invalid dates.
    data = data.dropna(
        subset=["date"]
    )

    # Remove duplicate dates.
    data = data.drop_duplicates(
        subset=["date"]
    )

    # Sort chronologically.
    data = data.sort_values(
        "date"
    )

    # Reset index.
    data = data.reset_index(
        drop=True
    )

    # Ensure close is numeric.
    data["close"] = pd.to_numeric(
        data["close"],
        errors="coerce"
    )

    # Remove rows without closing prices.
    data = data.dropna(
        subset=["close"]
    )

    # Remove invalid prices.
    data = data[
        data["close"] > 0
    ]

    if len(data) < 50:

        raise ValueError(
            "Dataset contains fewer than 50 valid observations. "
            "More historical data is recommended."
        )

    print("Dataset loaded successfully.")

    print("Dataset shape:", data.shape)

    return data


# ============================================================
# 2. CREATE EDA FEATURES
# ============================================================

def create_eda_features(data):

    print("\nCreating EDA features...")

    data = data.copy()

    # Daily percentage returns.
    data["daily_return"] = (
        data["close"].pct_change() * 100
    )

    # Moving averages.
    data["ma_5"] = (
        data["close"]
        .rolling(window=5)
        .mean()
    )

    data["ma_20"] = (
        data["close"]
        .rolling(window=20)
        .mean()
    )

    data["ma_50"] = (
        data["close"]
        .rolling(window=50)
        .mean()
    )

    # Rolling volatility.
    data["rolling_volatility"] = (
        data["daily_return"]
        .rolling(window=20)
        .std()
    )

    # Daily absolute price change.
    data["price_change"] = (
        data["close"].diff()
    )

    return data


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

def setup_directories():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )


# ============================================================
# 4. GRAPH 1: HISTORICAL PRICE TREND
# ============================================================

def plot_price_trend(data):

    print("Generating price trend graph...")

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        data["date"],
        data["close"],
        label="Closing Price",
        linewidth=1.5
    )

    plt.title(
        "Historical Sugar Price Trend",
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
        f"{OUTPUT_DIR}/01_price_trend.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 5. GRAPH 2: DAILY RETURNS
# ============================================================

def plot_daily_returns(data):

    print("Generating daily returns graph...")

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        data["date"],
        data["daily_return"],
        label="Daily Return",
        linewidth=0.8
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        "Daily Sugar Price Returns",
        fontsize=16
    )

    plt.xlabel("Date")

    plt.ylabel("Daily Return (%)")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/02_daily_returns.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 6. GRAPH 3: RETURN DISTRIBUTION
# ============================================================

def plot_return_distribution(data):

    print("Generating return distribution graph...")

    returns = data[
        "daily_return"
    ].dropna()

    plt.figure(
        figsize=(10, 6)
    )

    sns.histplot(
        returns,
        bins=50,
        kde=True
    )

    plt.title(
        "Distribution of Daily Returns",
        fontsize=16
    )

    plt.xlabel(
        "Daily Return (%)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/03_return_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 7. GRAPH 4: ROLLING VOLATILITY
# ============================================================

def plot_rolling_volatility(data):

    print("Generating rolling volatility graph...")

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        data["date"],
        data["rolling_volatility"],
        label="20-Day Rolling Volatility"
    )

    plt.title(
        "Sugar Price Rolling Volatility",
        fontsize=16
    )

    plt.xlabel("Date")

    plt.ylabel(
        "20-Day Standard Deviation of Returns (%)"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/04_rolling_volatility.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 8. GRAPH 5: CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(data):

    print("Generating correlation heatmap...")

    numeric_data = data.select_dtypes(
        include=np.number
    )

    correlation = numeric_data.corr()

    plt.figure(
        figsize=(12, 9)
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        center=0
    )

    plt.title(
        "Correlation Heatmap",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/05_correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. GRAPH 6: CLOSING PRICE BOXPLOT
# ============================================================

def plot_price_boxplot(data):

    print("Generating closing price boxplot...")

    plt.figure(
        figsize=(10, 6)
    )

    sns.boxplot(
        x=data["close"]
    )

    plt.title(
        "Distribution of Closing Prices",
        fontsize=16
    )

    plt.xlabel(
        "Closing Price"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/06_price_boxplot.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. GRAPH 7: MOVING AVERAGES
# ============================================================

def plot_moving_averages(data):

    print("Generating moving averages graph...")

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        data["date"],
        data["close"],
        label="Closing Price",
        alpha=0.7
    )

    plt.plot(
        data["date"],
        data["ma_5"],
        label="5-Day Moving Average"
    )

    plt.plot(
        data["date"],
        data["ma_20"],
        label="20-Day Moving Average"
    )

    plt.plot(
        data["date"],
        data["ma_50"],
        label="50-Day Moving Average"
    )

    plt.title(
        "Sugar Price and Moving Averages",
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
        f"{OUTPUT_DIR}/07_moving_averages.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 11. GENERATE STATISTICAL SUMMARY
# ============================================================

def generate_summary(data):

    print("\nGenerating statistical summary...")

    numeric_data = data.select_dtypes(
        include=np.number
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "COMMODITY PRICE EXPLORATORY DATA ANALYSIS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Dataset shape: {data.shape}\n\n"
        )

        file.write(
            f"Start date: {data['date'].min()}\n"
        )

        file.write(
            f"End date: {data['date'].max()}\n\n"
        )

        file.write(
            "STATISTICAL SUMMARY\n"
        )

        file.write(
            numeric_data.describe().to_string()
        )

        file.write(
            "\n\nMISSING VALUES\n"
        )

        file.write(
            data.isnull().sum().to_string()
        )

        file.write(
            "\n\nDAILY RETURN STATISTICS\n"
        )

        file.write(
            data["daily_return"]
            .describe()
            .to_string()
        )

        file.write(
            "\n\nCORRELATION MATRIX\n"
        )

        file.write(
            numeric_data.corr().to_string()
        )

        file.write(
            "\n\nPRICE RANGE\n"
        )

        file.write(
            f"Minimum closing price: "
            f"{data['close'].min():.4f}\n"
        )

        file.write(
            f"Maximum closing price: "
            f"{data['close'].max():.4f}\n"
        )

        file.write(
            f"Average closing price: "
            f"{data['close'].mean():.4f}\n"
        )

        file.write(
            f"Median closing price: "
            f"{data['close'].median():.4f}\n"
        )


# ============================================================
# 12. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)

    print(
        "COMMODITY PRICE EXPLORATORY DATA ANALYSIS"
    )

    print("=" * 60)

    setup_directories()

    # Load dataset.
    data = load_data()

    # Create features.
    data = create_eda_features(data)

    # Save EDA dataset.
    data.to_csv(
        EDA_DATA_PATH,
        index=False
    )

    # Display dataset information.
    print("\nFirst 5 rows:")

    print(data.head())

    print("\nStatistical summary:")

    print(data.describe())

    print("\nMissing values:")

    print(data.isnull().sum())

    # Generate graphs.
    plot_price_trend(data)

    plot_daily_returns(data)

    plot_return_distribution(data)

    plot_rolling_volatility(data)

    plot_correlation_heatmap(data)

    plot_price_boxplot(data)

    plot_moving_averages(data)

    # Generate summary.
    generate_summary(data)

    print("\n" + "=" * 60)

    print(
        "EDA COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    print(
        f"\nGraphs saved in: {OUTPUT_DIR}"
    )

    print(
        f"Summary saved in: {SUMMARY_PATH}"
    )

    print(
        f"EDA dataset saved in: {EDA_DATA_PATH}"
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
