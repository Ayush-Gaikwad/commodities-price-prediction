
import json
import random
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from fastai.tabular.all import *

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

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# --------------------------------------------------
# FASTAI DATASET
# --------------------------------------------------

train_df = train_df[FEATURES + ["target"]].copy()
test_df = test_df[FEATURES + ["target"]].copy()

# fastai tabular regression
data = TabularPandas(
    train_df,
    procs=[Normalize],
    cat_names=[],
    cont_names=FEATURES,
    y_names="target",
    y_block=RegressionBlock(),
    splits=RandomSplitter(valid_pct=0.1, seed=SEED)(
        range(len(train_df))
    )
)

dls = data.dataloaders(
    bs=64
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

learn = tabular_learner(
    dls,
    layers=[64, 32],
    metrics=mae
)

print("=" * 60)
print("FASTAI TABULAR REGRESSION")
print("=" * 60)

learn.fit_one_cycle(
    30,
    1e-2
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

test_data = test_df[FEATURES].copy()

test_dl = learn.dls.test_dl(test_data)

predictions, _ = learn.get_preds(dl=test_dl)

y_pred = predictions.numpy().reshape(-1)

y_true = test_df["target"].to_numpy()

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

mae_value = mean_absolute_error(y_true, y_pred)
rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
r2_value = r2_score(y_true, y_pred)

directional_accuracy = np.mean(
    np.sign(y_true) == np.sign(y_pred)
) * 100

print("\nFASTAI RESULTS")
print("MAE:", round(mae_value, 6))
print("RMSE:", round(rmse_value, 6))
print("R2:", round(r2_value, 6))
print("Directional Accuracy:", round(directional_accuracy, 2), "%")

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

learn.export(MODEL_DIR / "fastai_regression.pkl")

# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

result = pd.DataFrame({
    "date": df["date"].iloc[split_index:].reset_index(drop=True),
    "actual": y_true,
    "fastai_prediction": y_pred
})

result.to_csv(
    RESULT_DIR / "fastai_predictions.csv",
    index=False
)

metrics = pd.DataFrame([{
    "Model": "fastai Tabular Regression",
    "MAE": mae_value,
    "RMSE": rmse_value,
    "R2": r2_value,
    "Directional Accuracy": directional_accuracy
}])

metrics.to_csv(
    RESULT_DIR / "fastai_metrics.csv",
    index=False
)

print("\nFASTAI TRAINING COMPLETED")
print("Model saved:", MODEL_DIR / "fastai_regression.pkl")
print("Predictions saved:", RESULT_DIR / "fastai_predictions.csv")
