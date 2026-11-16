
import json
import random
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_FILE = Path("data/processed/sugar_features.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

with open(MODEL_DIR / "feature_columns.json", "r") as f:
    FEATURES = json.load(f)

X = df[FEATURES].astype(np.float32)
y = df["target"].astype(np.float32)

# --------------------------------------------------
# CHRONOLOGICAL SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = df["date"].iloc[split_index:].reset_index(drop=True)

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
y_test_np = y_test.to_numpy(dtype=np.float32)

# --------------------------------------------------
# TABNET MODEL
# --------------------------------------------------

print("=" * 65)
print("PYTORCH TABNET REGRESSION")
print("=" * 65)

model = TabNetRegressor(
    n_d=16,
    n_a=16,
    n_steps=3,
    gamma=1.3,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 0.02},
    mask_type="entmax",
    seed=SEED,
    verbose=10
)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

model.fit(
    X_train=X_train_scaled,
    y_train=y_train_np,
    eval_set=[
        (X_train_scaled, y_train_np)
    ],
    eval_name=["train"],
    eval_metric=["rmse"],
    max_epochs=100,
    patience=15,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

y_pred = model.predict(X_test_scaled).reshape(-1)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

mae_value = mean_absolute_error(y_test_np, y_pred)
rmse_value = np.sqrt(mean_squared_error(y_test_np, y_pred))
r2_value = r2_score(y_test_np, y_pred)

directional_accuracy = np.mean(
    np.sign(y_test_np) == np.sign(y_pred)
) * 100

print("\nTABNET RESULTS")
print(f"MAE: {mae_value:.6f}")
print(f"RMSE: {rmse_value:.6f}")
print(f"R2: {r2_value:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# --------------------------------------------------
# SAVE MODEL AND SCALER
# --------------------------------------------------

model.save_model(
    str(MODEL_DIR / "tabnet_regression")
)

import joblib

joblib.dump(
    scaler,
    MODEL_DIR / "tabnet_scaler.pkl"
)

# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

predictions = pd.DataFrame({
    "date": test_dates,
    "actual": y_test_np,
    "tabnet_prediction": y_pred
})

predictions.to_csv(
    RESULT_DIR / "tabnet_predictions.csv",
    index=False
)

metrics = pd.DataFrame([{
    "Model": "PyTorch TabNet",
    "MAE": mae_value,
    "RMSE": rmse_value,
    "R2": r2_value,
    "Directional Accuracy": directional_accuracy
}])

metrics.to_csv(
    RESULT_DIR / "tabnet_metrics.csv",
    index=False
)

print("\nTABNET TRAINING COMPLETED")
print("Model saved in:", MODEL_DIR)
print("Predictions saved in:", RESULT_DIR)
