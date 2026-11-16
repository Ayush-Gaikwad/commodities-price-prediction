
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

# --------------------------------------------------
# METRIC FILES
# --------------------------------------------------

metric_files = [
    "model_comparison.csv",
    "deep_learning_metrics.csv",
    "fastai_metrics.csv",
    "tabnet_metrics.csv",
    "flash_metrics.csv"
]

all_results = []

# --------------------------------------------------
# LOAD AND FILTER RESULTS
# --------------------------------------------------

for filename in metric_files:

    file_path = RESULTS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)

    if filename == "model_comparison.csv":
        # Keep only XGBoost from the original model comparison.
        df = df[df["Model"] == "XGBoost"]

    all_results.append(df)

# --------------------------------------------------
# COMBINE RESULTS
# --------------------------------------------------

comparison = pd.concat(
    all_results,
    ignore_index=True
)

# Keep only the five required frameworks.
required_models = [
    "XGBoost",
    "Keras Neural Network",
    "fastai Tabular Regression",
    "PyTorch TabNet",
    "Lightning Flash"
]

comparison = comparison[
    comparison["Model"].isin(required_models)
].copy()

# Ensure the requested order.
comparison["Model"] = pd.Categorical(
    comparison["Model"],
    categories=required_models,
    ordered=True
)

comparison = comparison.sort_values("Model")

comparison["Model"] = comparison["Model"].astype(str)

# --------------------------------------------------
# SAVE COMPARISON
# --------------------------------------------------

output_path = RESULTS_DIR / "framework_comparison.csv"

comparison.to_csv(output_path, index=False)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 75)
print("FIVE FRAMEWORK PERFORMANCE COMPARISON")
print("=" * 75)

print(comparison.to_string(index=False))

print("\nComparison saved successfully!")
print(output_path)

