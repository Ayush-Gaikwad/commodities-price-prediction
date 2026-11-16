
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from flash.core.model import Task
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "processed" / "sugar_features.csv"
FEATURE_PATH = PROJECT_DIR / "models" / "feature_columns.json"

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "flash_regression.ckpt"
SCALER_PATH = MODEL_DIR / "flash_scaler.npz"

PREDICTIONS_PATH = RESULTS_DIR / "flash_predictions.csv"
METRICS_PATH = RESULTS_DIR / "flash_metrics.csv"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
RANDOM_SEED = 42

pl.seed_everything(RANDOM_SEED, workers=True)


# --------------------------------------------------
# 1. NEURAL NETWORK
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


# --------------------------------------------------
# 2. LIGHTNING FLASH TASK
# --------------------------------------------------

class FlashRegressionTask(Task):

    def __init__(self, input_size):

        model = FlashRegressionNetwork(input_size)

        super().__init__(
            model=model,
            loss_fn=nn.MSELoss(),
            learning_rate=LEARNING_RATE
        )

    def training_step(self, batch, batch_idx):

        x, y = batch

        predictions = self(x)

        loss = nn.functional.mse_loss(predictions, y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        predictions = self(x)

        loss = nn.functional.mse_loss(predictions, y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return loss


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

with open(FEATURE_PATH, "r") as file:
    FEATURES = json.load(file)

# Identify the target column automatically.
# It must be the only column other than features and date.

excluded_columns = set(FEATURES + ["date"])

target_candidates = [
    column for column in df.columns
    if column not in excluded_columns
]

if len(target_candidates) != 1:
    raise ValueError(
        f"Expected exactly one target column, "
        f"but found: {target_candidates}"
    )

TARGET = target_candidates[0]

print("Target column:", TARGET)
print("Features:", FEATURES)
print("Dataset shape:", df.shape)

df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32).reshape(-1, 1)

dates = df["date"].values if "date" in df.columns else np.arange(len(df))


# --------------------------------------------------
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

split_index = int(len(X) * 0.80)

X_train_full = X[:split_index]
y_train_full = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

test_dates = dates[split_index:]

# Split training data into training and validation sets.
validation_index = int(len(X_train_full) * 0.90)

X_train = X_train_full[:validation_index]
y_train = y_train_full[:validation_index]

X_val = X_train_full[validation_index:]
y_val = y_train_full[validation_index:]

print("\nData split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. FEATURE STANDARDIZATION
# --------------------------------------------------

# Fit scaler using training data only.

feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)

feature_std[feature_std == 0] = 1.0

X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

np.savez(
    SCALER_PATH,
    mean=feature_mean,
    std=feature_std,
    features=np.array(FEATURES)
)

print("\nFeature standardization completed.")


# --------------------------------------------------
# 6. CREATE PYTORCH DATASETS
# --------------------------------------------------

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 7. INITIALIZE FLASH TASK
# --------------------------------------------------

print("\nInitializing Lightning Flash model...")

task = FlashRegressionTask(
    input_size=len(FEATURES)
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    mode="min"
)

checkpoint_callback = ModelCheckpoint(
    dirpath=str(MODEL_DIR),
    filename="flash_regression",
    monitor="val_loss",
    mode="min",
    save_top_k=1
)


# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="cpu",
    devices=1,
    callbacks=[
        early_stopping,
        checkpoint_callback
    ],
    logger=False,
    enable_progress_bar=True,
    enable_model_summary=False
)

print("\nStarting Lightning Flash training...\n")

trainer.fit(
    task,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)

# Save final training checkpoint.
trainer.save_checkpoint(str(MODEL_PATH))

print("\nTraining completed.")
print("Checkpoint saved:", MODEL_PATH)


# --------------------------------------------------
# 9. GENERATE TEST PREDICTIONS
# --------------------------------------------------

print("\nGenerating test predictions...")

task.eval()

predictions = []

with torch.no_grad():

    for start in range(0, len(X_test_tensor), BATCH_SIZE):

        batch = X_test_tensor[start:start + BATCH_SIZE]

        batch_predictions = task(batch)

        predictions.extend(
            batch_predictions.detach().cpu().numpy().reshape(-1)
        )

predictions = np.array(predictions)

actual = y_test.reshape(-1)


# --------------------------------------------------
# 10. EVALUATE MODEL
# --------------------------------------------------

mae = mean_absolute_error(actual, predictions)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

r2 = r2_score(actual, predictions)

directional_accuracy = (
    np.mean(np.sign(actual) == np.sign(predictions)) * 100
)

metrics = pd.DataFrame([
    {
        "Model": "Lightning Flash",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy
    }
])

metrics.to_csv(METRICS_PATH, index=False)


# --------------------------------------------------
# 11. SAVE PREDICTIONS
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "Date": test_dates,
    "Actual_Return": actual,
    "Predicted_Return": predictions
})

prediction_results.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# --------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("LIGHTNING FLASH MODEL RESULTS")
print("=" * 50)

print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2 Score: {r2:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

print("\nSaved files:")
print(MODEL_PATH)
print(SCALER_PATH)
print(PREDICTIONS_PATH)
print(METRICS_PATH)

print("\nLightning Flash training and evaluation completed!")
