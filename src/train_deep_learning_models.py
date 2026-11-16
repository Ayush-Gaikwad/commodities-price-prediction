
import json
import random
import numpy as np
import pandas as pd
import torch
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)

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

with open(MODEL_DIR / "feature_columns.json", "r") as f:
    FEATURES = json.load(f)

X = df[FEATURES].astype(np.float32)
y = df["target"].astype(np.float32)

# --------------------------------------------------
# CHRONOLOGICAL 80:20 SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = df["date"].iloc[split_index:].reset_index(drop=True)

# --------------------------------------------------
# SCALE FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.float32)

print("=" * 65)
print("DEEP LEARNING MODEL TRAINING")
print("=" * 65)

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Features:", len(FEATURES))

# --------------------------------------------------
# METRICS FUNCTION
# --------------------------------------------------

def evaluate_model(name, actual, predicted):

    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    directional_accuracy = np.mean(
        np.sign(actual) == np.sign(predicted)
    ) * 100

    print(f"\n{name}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy
    }


# --------------------------------------------------
# KERAS NEURAL NETWORK REGRESSION
# --------------------------------------------------

print("\nTraining Keras Neural Network...")

keras_model = keras.Sequential([
    layers.Input(shape=(len(FEATURES),)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

keras_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

keras_model.fit(
    X_train_scaled,
    y_train_np,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)

keras_predictions = keras_model.predict(
    X_test_scaled,
    verbose=0
).reshape(-1)

keras_model.save(MODEL_DIR / "keras_regression.keras")

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

predictions = pd.DataFrame({
    "date": test_dates,
    "actual": y_test_np,
    "keras_prediction": keras_predictions
})

metrics = [
    evaluate_model(
        "Keras Neural Network",
        y_test_np,
        keras_predictions
    )
]

predictions.to_csv(
    RESULT_DIR / "deep_learning_predictions.csv",
    index=False
)

pd.DataFrame(metrics).to_csv(
    RESULT_DIR / "deep_learning_metrics.csv",
    index=False
)

print("\nKeras training completed.")
print("Model saved:", MODEL_DIR / "keras_regression.keras")
print("Predictions saved:", RESULT_DIR / "deep_learning_predictions.csv")
