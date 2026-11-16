
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/sugar_prices.csv")
OUTPUT_FILE = Path("data/processed/sugar_features.csv")

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Daily percentage return
df["daily_return"] = df["close"].pct_change() * 100

# Lagged returns
for lag in [1, 2, 3, 5, 10]:
    df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)

# Moving-average ratios
for window in [5, 20]:
    moving_average = df["close"].rolling(window=window).mean()
    df[f"ma_ratio_{window}"] = (
        (df["close"] / moving_average) - 1
    ) * 100

# Rolling volatility
for window in [5, 20]:
    df[f"volatility_{window}"] = (
        df["daily_return"].rolling(window=window).std()
    )

# Volume change
df["volume_change"] = df["volume"].pct_change() * 100

# Next-day percentage return (target)
df["target"] = (
    (df["close"].shift(-1) - df["close"]) / df["close"]
) * 100

# Selected 11 features
FEATURES = [
    "daily_return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10",
    "ma_ratio_5",
    "ma_ratio_20",
    "volatility_5",
    "volatility_20",
    "volume_change"
]

# Keep only required columns
df = df[["date"] + FEATURES + ["target"]]

# Remove invalid values
df = df.replace([float("inf"), float("-inf")], pd.NA)
df = df.dropna().reset_index(drop=True)

# Save dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Number of features: {len(FEATURES)}")
print(f"Features: {FEATURES}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

print("\nTarget statistics:")
print(df["target"].describe())

print(f"\nSaved to: {OUTPUT_FILE}")
