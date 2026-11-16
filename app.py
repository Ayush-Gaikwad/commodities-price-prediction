
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torch import nn

st.set_page_config(
    page_title="Commodity Price Prediction",
    page_icon="📈",
    layout="wide"
)

RAW_DATA_PATH = "data/processed/sugar_prices.csv"
DATA_PATH = "data/processed/sugar_features.csv"
RESULTS_DIR = "results"
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

CLASSICAL_MODELS = {
    "Linear Regression": "linear_regression.pkl",
    "Ridge Regression": "ridge_regression.pkl",
    "ElasticNet": "elasticnet.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

MODEL_NAMES = [
    "Linear Regression",
    "Ridge Regression",
    "ElasticNet",
    "Random Forest",
    "XGBoost",
    "Keras Neural Network",
    "fastai Tabular Regression",
    "PyTorch TabNet",
    "Lightning Flash"
]

PREDICTION_FILES = {
    "Keras Neural Network": (
        "deep_learning_predictions.csv", "keras_prediction"
    ),
    "fastai Tabular Regression": (
        "fastai_predictions.csv", "fastai_prediction"
    ),
    "PyTorch TabNet": (
        "tabnet_predictions.csv", "tabnet_prediction"
    ),
    "Lightning Flash": (
        "flash_predictions.csv", "flash_prediction"
    )
}

METRIC_FILES = [
    "model_comparison.csv",
    "deep_learning_metrics.csv",
    "fastai_metrics.csv",
    "tabnet_metrics.csv",
    "flash_metrics.csv"
]


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_raw_data():
    raw = pd.read_csv(RAW_DATA_PATH)
    raw.columns = raw.columns.str.strip().str.lower()
    raw["date"] = pd.to_datetime(raw["date"])
    return raw.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_feature_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def create_latest_features(raw):
    data = raw.copy()
    data.columns = data.columns.str.strip().str.lower()
    data = data.sort_values("date").reset_index(drop=True)

    data["daily_return"] = data["close"].pct_change() * 100

    for lag in [1, 2, 3, 5, 10]:
        data[f"return_lag_{lag}"] = data["daily_return"].shift(lag)

    for window in [5, 20]:
        moving_average = data["close"].rolling(window).mean()
        data[f"ma_ratio_{window}"] = (
            (data["close"] / moving_average) - 1
        ) * 100

    for window in [5, 20]:
        data[f"volatility_{window}"] = (
            data["daily_return"].rolling(window).std()
        )

    data["volume_change"] = data["volume"].pct_change() * 100
    data = data.replace([np.inf, -np.inf], np.nan)

    latest = data.iloc[[-1]][FEATURE_COLUMNS]

    if latest.isnull().any().any():
        raise ValueError("Latest feature row contains missing values.")

    return latest


# --------------------------------------------------
# LOAD CLASSICAL MODELS
# --------------------------------------------------

@st.cache_resource
def load_classical_models():
    loaded = {}

    for name, filename in CLASSICAL_MODELS.items():
        path = os.path.join(MODEL_DIR, filename)
        loaded[name] = joblib.load(path)

    return loaded


# --------------------------------------------------
# LOAD KERAS MODEL AND RECREATE ITS SCALER
# --------------------------------------------------

@st.cache_resource
def load_keras_model():
    from tensorflow import keras

    model = keras.models.load_model(
        os.path.join(MODEL_DIR, "keras_regression.keras")
    )

    # The training script fitted StandardScaler on the first
    # chronological 80% of the feature dataset.
    from sklearn.preprocessing import StandardScaler

    data = load_feature_data()
    split_index = int(len(data) * 0.80)

    scaler = StandardScaler()
    scaler.fit(data.iloc[:split_index][FEATURE_COLUMNS])

    return model, scaler


# --------------------------------------------------
# LOAD FASTAI MODEL
# --------------------------------------------------

@st.cache_resource
def load_fastai_model():
    from fastai.tabular.all import load_learner

    return load_learner(
        os.path.join(MODEL_DIR, "fastai_regression.pkl")
    )


# --------------------------------------------------
# LOAD TABNET MODEL
# --------------------------------------------------

@st.cache_resource
def load_tabnet_model():
    from pytorch_tabnet.tab_model import TabNetRegressor

    model = TabNetRegressor()
    model.load_model(
        os.path.join(MODEL_DIR, "tabnet_regression.zip")
    )

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "tabnet_scaler.pkl")
    )

    return model, scaler


# --------------------------------------------------
# LOAD LIGHTNING FLASH CHECKPOINT
# --------------------------------------------------

class FlashRegressionNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_flash_model():
    checkpoint_path = os.path.join(
        MODEL_DIR, "flash_regression.ckpt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = FlashRegressionNetwork(len(FEATURE_COLUMNS))

    state_dict = {
        key.replace("model.", "", 1): value
        for key, value in checkpoint["state_dict"].items()
    }

    model.load_state_dict(state_dict)
    model.eval()

    scaler_data = np.load(
        os.path.join(MODEL_DIR, "flash_scaler.npz")
    )

    mean = scaler_data["mean"]
    std = scaler_data["std"]

    return model, mean, std


# --------------------------------------------------
# MAKE PREDICTIONS USING ALL NINE MODELS
# --------------------------------------------------

def predict_all_models(features, classical_models):
    results = {}

    for name, model in classical_models.items():
        results[name] = float(
            model.predict(features)[0]
        )

    keras_model, keras_scaler = load_keras_model()
    keras_input = keras_scaler.transform(features)
    results["Keras Neural Network"] = float(
        keras_model.predict(
            keras_input, verbose=0
        ).reshape(-1)[0]
    )

    fastai_model = load_fastai_model()
    fastai_dl = fastai_model.dls.test_dl(features)
    fastai_prediction, _ = fastai_model.get_preds(dl=fastai_dl)
    results["fastai Tabular Regression"] = float(
        fastai_prediction.numpy().reshape(-1)[0]
    )

    tabnet_model, tabnet_scaler = load_tabnet_model()
    tabnet_input = tabnet_scaler.transform(features)
    results["PyTorch TabNet"] = float(
        tabnet_model.predict(tabnet_input).reshape(-1)[0]
    )

    flash_model, flash_mean, flash_std = load_flash_model()
    flash_input = (
        features.to_numpy(dtype=np.float32) - flash_mean
    ) / flash_std

    with torch.no_grad():
        flash_tensor = torch.tensor(
            flash_input, dtype=torch.float32
        )
        results["Lightning Flash"] = float(
            flash_model(flash_tensor).item()
        )

    return results


# --------------------------------------------------
# LOAD METRICS
# --------------------------------------------------

@st.cache_data
def load_all_metrics():
    frames = []

    for filename in METRIC_FILES:
        path = os.path.join(RESULTS_DIR, filename)

        if os.path.exists(path):
            frame = pd.read_csv(path)

            if "Model" in frame.columns:
                frames.append(frame)

    if not frames:
        return pd.DataFrame()

    metrics = pd.concat(frames, ignore_index=True)

    # Avoid duplicate XGBoost rows from framework comparison.
    metrics = metrics.drop_duplicates(
        subset=["Model"], keep="first"
    )

    return metrics


# --------------------------------------------------
# LOAD PREDICTION HISTORY
# --------------------------------------------------

def standardize_prediction_file(path, prediction_column):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()

    lower_map = {
        column.lower(): column for column in frame.columns
    }

    date_col = lower_map.get("date") or lower_map.get("target_date")
    actual_col = lower_map.get("actual")
    pred_col = lower_map.get(prediction_column.lower())

    # Support alternate column names in the Flash output.
    if pred_col is None:
        for column in frame.columns:
            if "predicted" in column.lower():
                pred_col = column
                break

    if actual_col is None:
        for column in frame.columns:
            if column.lower() in ["actual_return", "y_true"]:
                actual_col = column
                break

    if date_col is None or actual_col is None or pred_col is None:
        return None

    result = pd.DataFrame({
        "date": pd.to_datetime(frame[date_col]),
        "actual": pd.to_numeric(frame[actual_col], errors="coerce"),
        "prediction": pd.to_numeric(frame[pred_col], errors="coerce")
    })

    return result.dropna().sort_values("date")


@st.cache_data
def load_prediction_history():
    history = {}

    # Classical models are stored together in predictions.csv.
    classical_path = os.path.join(RESULTS_DIR, "predictions.csv")

    if os.path.exists(classical_path):
        frame = pd.read_csv(classical_path)
        frame.columns = frame.columns.str.strip()

        lower_map = {
            column.lower(): column for column in frame.columns
        }

        date_col = (
            lower_map.get("date")
            or lower_map.get("target_date")
        )
        actual_col = lower_map.get("actual")

        if date_col and actual_col:
            for name in CLASSICAL_MODELS:
                pred_col = lower_map.get(
                    name.lower().replace(" ", "_") + "_prediction"
                )

                if pred_col:
                    history[name] = pd.DataFrame({
                        "date": pd.to_datetime(frame[date_col]),
                        "actual": pd.to_numeric(
                            frame[actual_col], errors="coerce"
                        ),
                        "prediction": pd.to_numeric(
                            frame[pred_col], errors="coerce"
                        )
                    }).dropna().sort_values("date")

    # Framework model prediction files.
    for name, (filename, column) in PREDICTION_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)

        if os.path.exists(path):
            result = standardize_prediction_file(path, column)

            if result is not None:
                history[name] = result

    return history


# --------------------------------------------------
# CHECK REQUIRED FILES
# --------------------------------------------------

required_files = [
    RAW_DATA_PATH,
    DATA_PATH,
    os.path.join(MODEL_DIR, "feature_columns.json")
]

required_files += [
    os.path.join(MODEL_DIR, filename)
    for filename in CLASSICAL_MODELS.values()
]

required_files += [
    os.path.join(MODEL_DIR, "keras_regression.keras"),
    os.path.join(MODEL_DIR, "fastai_regression.pkl"),
    os.path.join(MODEL_DIR, "tabnet_regression.zip"),
    os.path.join(MODEL_DIR, "tabnet_scaler.pkl"),
    os.path.join(MODEL_DIR, "flash_regression.ckpt"),
    os.path.join(MODEL_DIR, "flash_scaler.npz")
]

missing = [path for path in required_files if not os.path.exists(path)]

if missing:
    st.error("Missing required files:")
    for path in missing:
        st.write(path)
    st.stop()


# --------------------------------------------------
# LOAD APPLICATION DATA
# --------------------------------------------------

try:
    raw_data = load_raw_data()
    data = load_feature_data()
    latest_features = create_latest_features(raw_data)
    classical_models = load_classical_models()
    metrics = load_all_metrics()
    prediction_history = load_prediction_history()

except Exception as error:
    st.error(f"Unable to load application data: {error}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📈 Commodity Price Prediction")
st.subheader("Sugar Return Forecasting Using Machine Learning")

st.markdown(
    """
    This dashboard compares nine regression models trained to
    predict the next trading day's percentage return of sugar futures.
    The implied closing price is calculated from the predicted return.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Application Settings")

selected_model = st.sidebar.selectbox(
    "Select Regression Model",
    MODEL_NAMES
)

st.sidebar.markdown(
    """
    ---
    **Project Information**

    Commodity: Sugar Futures

    Data Source: Yahoo Finance

    Target: Next-day percentage return

    Models: 9 regression models
    """
)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

st.header("Latest Prediction")

try:
    with st.spinner("Generating predictions from all nine models..."):
        all_predictions = predict_all_models(
            latest_features, classical_models
        )

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

latest_row = raw_data.iloc[-1]
latest_price = float(latest_row["close"])
latest_date = latest_row["date"]

predicted_return = all_predictions[selected_model]
implied_next_price = latest_price * (
    1 + predicted_return / 100
)

price_change = implied_next_price - latest_price

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Closing Price",
    f"{latest_price:.4f}"
)

col2.metric(
    "Predicted Next-Day Return",
    f"{predicted_return:+.4f}%"
)

col3.metric(
    "Implied Next Closing Price",
    f"{implied_next_price:.4f}",
    delta=f"{price_change:+.4f}"
)

st.caption(
    f"Latest available market data: {latest_date:%Y-%m-%d}"
)

st.caption(f"Selected model: {selected_model}")

st.warning(
    "These are experimental predictions, not guaranteed market prices "
    "or financial advice."
)

st.divider()


# --------------------------------------------------
# ALL MODEL PREDICTIONS
# --------------------------------------------------

st.header("Latest Predictions from All 9 Models")

prediction_table = pd.DataFrame({
    "Model": list(all_predictions.keys()),
    "Predicted Return (%)": list(all_predictions.values())
})

prediction_table["Implied Next Closing Price"] = (
    latest_price * (
        1 + prediction_table["Predicted Return (%)"] / 100
    )
)

st.dataframe(
    prediction_table,
    use_container_width=True,
    hide_index=True
)

st.divider()


# --------------------------------------------------
# HISTORICAL PRICE CHART
# --------------------------------------------------

st.header("Historical Sugar Prices")

historical = raw_data.tail(250).copy()
historical["ma_20"] = historical["close"].rolling(20).mean()
historical["ma_50"] = historical["close"].rolling(50).mean()

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    historical["date"],
    historical["close"],
    label="Closing Price",
    linewidth=2
)

ax.plot(
    historical["date"],
    historical["ma_20"],
    label="20-Day Moving Average",
    linestyle="--"
)

ax.plot(
    historical["date"],
    historical["ma_50"],
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


# --------------------------------------------------
# ACTUAL VS PREDICTED RETURNS
# --------------------------------------------------

st.header("Actual vs Predicted Next-Day Returns")

if selected_model in prediction_history:
    selected_history = prediction_history[selected_model]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        selected_history["date"],
        selected_history["actual"],
        label="Actual Return",
        linewidth=2
    )

    ax.plot(
        selected_history["date"],
        selected_history["prediction"],
        label=selected_model,
        alpha=0.8
    )

    ax.set_title(
        f"Actual vs Predicted Returns - {selected_model}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Next-Day Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info(
        "Historical prediction file or expected columns were not found "
        "for this model. The latest prediction is still available above."
    )

st.divider()


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.header("Model Performance Comparison")

if not metrics.empty:
    st.caption(
        "MAE and RMSE are measured in percentage points. "
        "Lower MAE/RMSE and higher R² are preferred."
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    selected_metrics = metrics[
        metrics["Model"] == selected_model
    ]

    if not selected_metrics.empty:
        row = selected_metrics.iloc[0]

        st.subheader(f"{selected_model} Performance")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("MAE", f"{row['MAE']:.4f}")
        c2.metric("RMSE", f"{row['RMSE']:.4f}")
        c3.metric("R² Score", f"{row['R2']:.4f}")
        c4.metric(
            "Directional Accuracy",
            f"{row['Directional Accuracy']:.2f}%"
        )

else:
    st.warning("No model metrics files were found.")

st.divider()


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.header("Dataset Preview")

st.dataframe(
    data.tail(10),
    use_container_width=True,
    hide_index=True
)

st.divider()


# --------------------------------------------------
# DOWNLOAD PREDICTIONS
# --------------------------------------------------

st.header("Download Latest Model Predictions")

download_data = prediction_table.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download All 9 Model Predictions",
    data=download_data,
    file_name="sugar_all_model_predictions.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "Academic ML Project: Commodity Price Prediction Using "
    "Machine Learning Regression."
)

st.caption(
    "Predictions are experimental model outputs and should not "
    "be treated as financial advice."
)
