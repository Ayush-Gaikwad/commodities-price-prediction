
# ============================================================
# PHASE 7: STREAMLIT WEB APPLICATION
# Commodities Price Prediction Using Machine Learning Regression
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

DATA_PATH = "data/processed/sugar_features.csv"

PREDICTIONS_PATH = "results/predictions.csv"

METRICS_PATH = "results/model_comparison.csv"

MODEL_DIR = "models"


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
    "open_close_diff"
]




# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data["target_date"] = pd.to_datetime(
        data["target_date"]
    )

    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    return data


@st.cache_data
def load_predictions():

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    predictions["target_date"] = pd.to_datetime(
        predictions["target_date"]
    )

    return predictions


@st.cache_data
def load_metrics():

    metrics = pd.read_csv(
        METRICS_PATH
    )

    return metrics


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    model_files = {

        "Linear Regression":
        "linear_regression.pkl",

        "Random Forest":
        "random_forest.pkl",

        "XGBoost":
        "xgboost.pkl"

    }

    for name, filename in model_files.items():

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        models[name] = joblib.load(
            path
        )

    return models


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "📈 Commodity Price Prediction"
)

st.subheader(
    "Sugar Price Forecasting Using Machine Learning Regression"
)

st.markdown(
    """
    This application analyzes historical sugar prices and uses
    trained machine learning regression models to forecast the
    next trading day's closing price.
    """
)

st.divider()


# ============================================================
# CHECK FILES
# ============================================================

required_files = [

    DATA_PATH,

    PREDICTIONS_PATH,

    METRICS_PATH,

    f"{MODEL_DIR}/linear_regression.pkl",

    f"{MODEL_DIR}/random_forest.pkl",

    f"{MODEL_DIR}/xgboost.pkl"

]


missing_files = [

    path

    for path in required_files

    if not os.path.exists(path)

]


if missing_files:

    st.error(
        "The following required files are missing:"
    )

    for path in missing_files:

        st.write(
            f"- {path}"
        )

    st.stop()


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

data = load_data()

predictions = load_predictions()

metrics = load_metrics()

models = load_models()

# Match the feature selection used during model training.
if "volume" in data.columns:
    FEATURE_COLUMNS.append("volume")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Application Settings"
)

selected_model = st.sidebar.selectbox(

    "Select Regression Model",

    [

        "Linear Regression",

        "Random Forest",

        "XGBoost"

    ]

)

st.sidebar.markdown(
    """
    ---
    **Project Information**

    Commodity: Sugar

    Task: Next-day closing price prediction

    Algorithms:
    - Linear Regression
    - Random Forest
    - XGBoost
    """
)


# ============================================================
# PRICE OVERVIEW
# ============================================================

latest_row = data.iloc[-1]

latest_price = float(
    latest_row["close"]
)

latest_date = latest_row["date"]

latest_prediction = float(
    models[selected_model].predict(

        data[FEATURE_COLUMNS].iloc[[-1]]

    )[0]
)


# ============================================================
# PRICE METRICS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(

        "Latest Closing Price",

        f"{latest_price:.4f}"

    )


with col2:

    st.metric(

        "Predicted Next Closing Price",

        f"{latest_prediction:.4f}",

        delta=round(

            latest_prediction - latest_price,

            4

        )

    )


with col3:

    st.metric(

        "Latest Available Date",

        latest_date.strftime(
            "%Y-%m-%d"
        )

    )


st.caption(
    "Price units are the same as those used in the source dataset."
)

st.divider()


# ============================================================
# HISTORICAL PRICE CHART
# ============================================================

st.header(
    "Historical Sugar Prices"
)

historical_data = data.tail(
    250
)

fig, ax = plt.subplots(
    figsize=(14, 6)
)

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

ax.set_title(
    "Historical Sugar Closing Prices"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Closing Price"
)

ax.legend()

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

st.pyplot(
    fig
)

plt.close(fig)


st.divider()


# ============================================================
# ACTUAL VS PREDICTED PRICES
# ============================================================

st.header(
    "Actual vs Predicted Prices"
)

prediction_column = (

    selected_model.lower()

    .replace(" ", "_")

    + "_prediction"

)


fig, ax = plt.subplots(
    figsize=(14, 6)
)

ax.plot(

    predictions["target_date"],

    predictions["actual"],

    label="Actual Price",

    linewidth=2

)

ax.plot(

    predictions["target_date"],

    predictions[prediction_column],

    label=selected_model,

    alpha=0.8

)

ax.set_title(
    f"Actual vs Predicted Prices - {selected_model}"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Closing Price"
)

ax.legend()

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

st.pyplot(
    fig
)

plt.close(fig)


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header(
    "Model Performance Comparison"
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

    selected_metrics = (
        selected_metrics.iloc[0]
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

            "MAPE",

            f"{selected_metrics['MAPE']:.4f}%"

        )


st.divider()


# ============================================================
# DATASET PREVIEW
# ============================================================

st.header(
    "Dataset Preview"
)

st.dataframe(

    data.tail(10),

    use_container_width=True

)


# ============================================================
# DOWNLOAD PREDICTIONS
# ============================================================

st.header(
    "Download Predictions"
)

csv_data = predictions.to_csv(
    index=False
).encode("utf-8")


st.download_button(

    label="Download Prediction Results",

    data=csv_data,

    file_name="sugar_price_predictions.csv",

    mime="text/csv"

)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(

    "Academic ML project: Commodity Price Prediction "
    "Using Machine Learning Regression."

)

st.caption(

    "Predictions are experimental model outputs and "
    "should not be treated as financial advice."

)
