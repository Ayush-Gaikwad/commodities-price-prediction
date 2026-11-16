
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD COMPARISON DATA
# --------------------------------------------------

df = pd.read_csv(
    RESULTS_DIR / "framework_comparison.csv"
)

# --------------------------------------------------
# GRAPH CONFIGURATION
# --------------------------------------------------

metrics = {
    "MAE": "Mean Absolute Error (MAE)",
    "RMSE": "Root Mean Squared Error (RMSE)",
    "R2": "R-squared (R²)",
    "Directional Accuracy": "Directional Accuracy (%)"
}

# --------------------------------------------------
# GENERATE GRAPHS
# --------------------------------------------------

for column, title in metrics.items():

    plt.figure(figsize=(10, 6))

    plt.bar(
        df["Model"],
        df[column]
    )

    plt.title(title)
    plt.xlabel("Framework")
    plt.ylabel(column)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    filename = column.lower().replace(" ", "_") + "_comparison.png"

    plt.savefig(
        GRAPH_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", GRAPH_DIR / filename)

print("\nAll four comparison graphs generated successfully!")
