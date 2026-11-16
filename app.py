
# ============================================================
# PHASE 7: STREAMLIT WEB APPLICATION
# Commodities Price Prediction Using Machine Learning Regression
# Target: Next-Day Percentage Return
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Commodity Price Prediction",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATA_PATH = "data/processed/sugar_prices.csv"
DATA_PATH = "data/processed/sugar_features.csv"

PREDICTIONS_PATH = "results/predictions.csv"
METRICS_PATH = "results/model_comparison.csv"

MODEL_DIR = "models"

FEATURE_COLUMNS = [
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

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Ridge Regression": "ridge_regression.pkl",
    "ElasticNet": "elasticnet.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_raw_data():

    raw = pd.read_csv(RAW_DATA_PATH)

    raw.columns = raw.columns.str.strip().str.lower()

    raw["date"] = pd.to_datetime(raw["date"])

    raw = raw.sort_values("date").reset_index(drop=True)

    return raw


@st.cache_data
def load_feature_data():

    data = pd.read_csv(DATA_PATH)

    data["date"] = pd.to_datetime(data["date"])

    return data.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_predictions():

    predictions = pd.read_csv(PREDICTIONS_PATH)

    # Support either date-column naming convention.
    if "date" in predictions.columns:
        predictions["date"] = pd.to_datetime(predictions["date"])
        date_column = "date"

    elif "target_date" in predictions.columns:
        predictions["target_date"] = pd.to_datetime(
            predictions["target_date"]
        )
        date_column = "target_date"

    else:
        raise ValueError(
            "Predictions file must contain date or target_date."
        )

    return predictions.sort_values(
        date_column
    ).reset_index(drop=True)


@st.cache_data
def load_metrics():

    return pd.read_csv(METRICS_PATH)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    for name, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        models[name] = joblib.load(path)

    return models


# ============================================================
# CREATE LATEST INFERENCE FEATURES
# Must match feature_engineering.py exactly.
# ============================================================

def create_latest_features(raw):

    data = raw.copy()

    data.columns = data.columns.str.strip().str.lower()

    data = data.sort_values("date").reset_index(drop=True)

    # Daily percentage return
    data["daily_return"] = (
        data["close"].pct_change() * 100
    )

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:

        data[f"return_lag_{lag}"] = (
            data["daily_return"].shift(lag)
        )

    # Moving-average ratios
    for window in [5, 20]:

        moving_average = (
            data["close"].rolling(window=window).mean()
        )

        data[f"ma_ratio_{window}"] = (
            (data["close"] / moving_average) - 1
        ) * 100

    # Rolling volatility
    for window in [5, 20]:

        data[f"volatility_{window}"] = (
            data["daily_return"]
            .rolling(window=window)
            .std()
        )

    # Volume change
    data["volume_change"] = (
        data["volume"].pct_change() * 100
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    latest_features = data.iloc[[-1]][FEATURE_COLUMNS]

    if latest_features.isnull().any().any():

        raise ValueError(
            "Latest row contains missing feature values. "
            "Check the raw dataset."
        )

    return latest_features


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    RAW_DATA_PATH,
    DATA_PATH,
    PREDICTIONS_PATH,
    METRICS_PATH
]

required_files += [
    os.path.join(MODEL_DIR, filename)
    for filename in MODEL_FILES.values()
]

missing_files = [
    path for path in required_files
    if not os.path.exists(path)
]

if missing_files:

    st.error("The following required files are missing:")

    for path in missing_files:
        st.write(f"- {path}")

    st.stop()


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

try:

    raw_data = load_raw_data()
    data = load_feature_data()
    predictions = load_predictions()
    metrics = load_metrics()
    models = load_models()

    latest_features = create_latest_features(raw_data)

except Exception as error:

    st.error(f"Unable to load application data: {error}")
    st.stop()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📈 Commodity Price Prediction")

st.subheader(
    "Sugar Return Forecasting Using Machine Learning Regression"
)

st.markdown(
    """
    This application analyzes historical sugar prices and uses
    five machine learning regression models to predict the
    next trading day's percentage return.

    The predicted return is also used to calculate an
    implied next closing price.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Application Settings")

selected_model = st.sidebar.selectbox(
    "Select Regression Model",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown(
    """
    ---

    **Project Information**

    Commodity: Sugar

    Prediction Target: Next-day percentage return

    Algorithms:
    - Linear Regression
    - Ridge Regression
    - ElasticNet
    - Random Forest
    - XGBoost

    Data Source: Yahoo Finance
    """
)


# ============================================================
# MAKE LATEST PREDICTION
# ============================================================

latest_row = raw_data.iloc[-1]

latest_price = float(latest_row["close"])

latest_date = latest_row["date"]

predicted_return = float(
    models[selected_model].predict(
        latest_features
    )[0]
)

implied_next_price = (
    latest_price * (1 + predicted_return / 100)
)

price_change = implied_next_price - latest_price


# ============================================================
# LATEST PREDICTION OVERVIEW
# ============================================================

st.header("Latest Prediction")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Latest Closing Price",
        f"{latest_price:.4f}"
    )

with col2:

    st.metric(
        "Predicted Next-Day Return",
        f"{predicted_return:+.4f}%"
    )

with col3:

    st.metric(
        "Implied Next Closing Price",
        f"{implied_next_price:.4f}",
        delta=f"{price_change:+.4f}"
    )

st.caption(
    f"Latest available market data: {latest_date.strftime('%Y-%m-%d')}"
)

st.caption(
    f"Selected model: {selected_model}"
)

st.warning(
    "These are experimental model predictions. "
    "The implied price is calculated from the predicted return "
    "and is not a guaranteed market price."
)

st.divider()


# ============================================================
# HISTORICAL PRICE CHART
# ============================================================

st.header("Historical Sugar Prices")

historical_data = raw_data.tail(250).copy()

historical_data["ma_20"] = (
    historical_data["close"].rolling(20).mean()
)

historical_data["ma_50"] = (
    historical_data["close"].rolling(50).mean()
)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    historical_data["date"],
    historical_data["close"],
    label="Closing Price",
    linewidth=2
)

ax.plot(
    historical_data["date"],
    historical_data["ma_20"],
    label="20-Day Moving Average",
    linestyle="--"
)

ax.plot(
    historical_data["date"],
    historical_data["ma_50"],
    label="50-Day Moving Average",
    linestyle=":"
)

ax.set_title("Historical Sugar Closing Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")

ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()

st.pyplot(fig)

plt.close(fig)

st.divider()


# ============================================================
# ACTUAL VS PREDICTED RETURNS
# ============================================================

st.header("Actual vs Predicted Next-Day Returns")

prediction_column = (
    selected_model.lower().replace(" ", "_")
    + "_prediction"
)

if prediction_column not in predictions.columns:

    st.error(
        f"Prediction column not found: {prediction_column}"
    )

else:

    date_column = (
        "date" if "date" in predictions.columns
        else "target_date"
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        predictions[date_column],
        predictions["actual"],
        label="Actual Return",
        linewidth=2
    )

    ax.plot(
        predictions[date_column],
        predictions[prediction_column],
        label=selected_model,
        alpha=0.8
    )

    ax.set_title(
        f"Actual vs Predicted Returns - {selected_model}"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return (%)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

st.divider()


# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

st.header("Model Performance Comparison")

st.caption(
    "MAE and RMSE are measured in percentage points. "
    "Lower values are better. Higher R² is better."
)

st.dataframe(
    metrics,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SELECTED MODEL METRICS
# ============================================================

selected_metrics = metrics[
    metrics["Model"] == selected_model
]

if not selected_metrics.empty:

    selected_metrics = selected_metrics.iloc[0]

    st.subheader(
        f"{selected_model} Performance"
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    with metric_col1:

        st.metric(
            "MAE",
            f"{selected_metrics['MAE']:.4f}"
        )

    with metric_col2:

        st.metric(
            "RMSE",
            f"{selected_metrics['RMSE']:.4f}"
        )

    with metric_col3:

        st.metric(
            "R² Score",
            f"{selected_metrics['R2']:.4f}"
        )

    with metric_col4:

        st.metric(
            "Directional Accuracy",
            f"{selected_metrics['Directional Accuracy']:.2f}%"
        )

else:

    st.info(
        "Performance metrics for this model are not available."
    )

st.divider()


# ============================================================
# DATASET PREVIEW
# ============================================================

st.header("Dataset Preview")

st.dataframe(
    data.tail(10),
    use_container_width=True
)

st.divider()


# ============================================================
# DOWNLOAD PREDICTIONS
# ============================================================

st.header("Download Prediction Results")

csv_data = predictions.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=csv_data,
    file_name="sugar_return_predictions.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Academic ML Project: Commodity Price Prediction "
    "Using Machine Learning Regression."
)

st.caption(
    "Predictions are experimental model outputs "
    "and should not be treated as financial advice."
)
