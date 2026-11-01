
import os
import sys

import pandas as pd
import yfinance as yf


# ==========================================
# CONFIGURATION
# ==========================================

TICKER = "SB=F"

START_DATE = "2015-01-01"

END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

RAW_PATH = "data/raw/sugar_prices_raw.csv"

PROCESSED_PATH = "data/processed/sugar_prices.csv"


# ==========================================
# DATA COLLECTION
# ==========================================

def download_data():

    print("=" * 60)
    print("SUGAR COMMODITY PRICE DATA COLLECTION")
    print("=" * 60)

    print(f"Ticker: {TICKER}")
    print(f"Start date: {START_DATE}")
    print(f"End date: {END_DATE}")

    try:

        data = yf.download(
            TICKER,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

    except Exception as error:

        print("\nError downloading data:")
        print(error)

        sys.exit(1)

    if data.empty:

        print("\nNo data downloaded.")

        print(
            "Check your internet connection "
            "or the availability of the ticker."
        )

        sys.exit(1)

    # Handle MultiIndex columns returned by yfinance.
    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    # Convert index into a regular column.
    data = data.reset_index()

    # Standardize column names.
    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]

    # Ensure the date column is named date.
    if "date" not in data.columns:

        if "datetime" in data.columns:

            data = data.rename(
                columns={"datetime": "date"}
            )

        else:

            print("Date column not found.")

            sys.exit(1)

    # Convert date column.
    data["date"] = pd.to_datetime(data["date"])

    # Remove duplicate dates.
    data = data.drop_duplicates(
        subset=["date"]
    )

    # Sort data chronologically.
    data = data.sort_values("date")

    # Save raw dataset.
    os.makedirs("data/raw", exist_ok=True)

    data.to_csv(
        RAW_PATH,
        index=False
    )

    print("\nRaw data downloaded successfully.")

    return data


# ==========================================
# DATA CLEANING
# ==========================================

def clean_data(data):

    print("\nCleaning the dataset...")

    data = data.copy()

    # Closing price is required for prediction.
    data = data.dropna(
        subset=["close"]
    )

    # Remove invalid closing prices.
    data = data[
        data["close"] > 0
    ]

    # Remove rows with invalid dates.
    data = data.dropna(
        subset=["date"]
    )

    # Remove duplicate dates again for safety.
    data = data.drop_duplicates(
        subset=["date"]
    )

    # Sort chronologically.
    data = data.sort_values("date")

    # Reset index.
    data = data.reset_index(drop=True)

    return data


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    # Download historical data.
    raw_data = download_data()

    # Clean the dataset.
    cleaned_data = clean_data(raw_data)

    # Create processed directory.
    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # Save cleaned data.
    cleaned_data.to_csv(
        PROCESSED_PATH,
        index=False
    )

    # Display results.
    print("\n" + "=" * 60)

    print("DATA COLLECTION COMPLETED")

    print("=" * 60)

    print(f"\nRaw rows: {len(raw_data)}")

    print(f"Cleaned rows: {len(cleaned_data)}")

    print("\nDataset shape:")

    print(cleaned_data.shape)

    print("\nColumn names:")

    print(cleaned_data.columns.tolist())

    print("\nFirst 5 rows:")

    print(cleaned_data.head())

    print("\nLast 5 rows:")

    print(cleaned_data.tail())

    print("\nMissing values:")

    print(cleaned_data.isnull().sum())

    print(
        f"\nRaw dataset saved to: {RAW_PATH}"
    )

    print(
        f"Cleaned dataset saved to: {PROCESSED_PATH}"
    )


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    main()
