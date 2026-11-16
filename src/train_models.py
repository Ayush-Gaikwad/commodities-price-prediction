
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_FILE = Path("data/processed/sugar_features.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

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

X = df[FEATURES]
y = df["target"]

# --------------------------------------------------
# CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

train_dates = df["date"].iloc[:split_index]
test_dates = df["date"].iloc[split_index:]

print("=" * 65)
print("11-FEATURE MODEL TRAINING")
print("=" * 65)

print(f"Total observations: {len(df)}")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")
print(f"Number of features: {len(FEATURES)}")
print(f"Training period: {train_dates.min()} to {train_dates.max()}")
print(f"Testing period: {test_dates.min()} to {test_dates.max()}")

# Save feature names
with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(FEATURES, f, indent=4)

# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

tscv = TimeSeriesSplit(n_splits=5)

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ]),
        {
            "model__alpha": [0.01, 0.1, 1, 10, 100]
        },
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    ),

    "ElasticNet": GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=20000))
        ]),
        {
            "model__alpha": [0.001, 0.01, 0.1, 1],
            "model__l1_ratio": [0.2, 0.5, 0.8, 1.0]
        },
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )
}

# --------------------------------------------------
# TRAIN AND SAVE MODELS
# --------------------------------------------------

predictions = pd.DataFrame({
    "date": test_dates.reset_index(drop=True),
    "actual": y_test.reset_index(drop=True)
})

for name, model in models.items():

    print("\n" + "-" * 65)
    print(f"Training {name}...")
    
    model.fit(X_train, y_train)

    if isinstance(model, GridSearchCV):
        print(f"Best parameters: {model.best_params_}")
        print(
            f"Best CV RMSE: "
            f"{-model.best_score_:.6f}"
        )

    y_pred = model.predict(X_test)

    column_name = (
        name.lower()
        .replace(" ", "_")
    ) + "_prediction"

    predictions[column_name] = y_pred

    model_filename = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(model, MODEL_DIR / model_filename)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Test RMSE: {rmse:.6f}")
    print(f"Saved model: {MODEL_DIR / model_filename}")

# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

predictions.to_csv(
    RESULT_DIR / "predictions.csv",
    index=False
)

print("\n" + "=" * 65)
print("MODEL TRAINING COMPLETED")
print("=" * 65)
print(f"Predictions saved to: {RESULT_DIR / 'predictions.csv'}")
print(f"Models saved to: {MODEL_DIR}")
