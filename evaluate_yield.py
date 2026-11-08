import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("data/yield/df_yield_merged.csv")
MODEL_PATH = Path("models/yield_model.joblib")
OUTPUT_DIR = Path("models")

key_columns = ["area", "year", "item"]
feature_columns = [
    "area",
    "year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "item",
]
target_column = "value_hg_ha"

data = pd.read_csv(DATA_PATH)
data = data.dropna(subset=feature_columns + [target_column])

# Match the duplicate handling used during training.
data = (
    data.groupby(key_columns, as_index=False)
    .agg(
        {
            "average_rain_fall_mm_per_year": "median",
            "pesticides_tonnes": "median",
            "avg_temp": "median",
            "value_hg_ha": "median",
        }
    )
)

test_data = data[data["year"] >= 2010]
X_test = test_data[feature_columns]
actual_yield = test_data[target_column].to_numpy()

yield_model = joblib.load(MODEL_PATH)
predicted_yield = yield_model.predict(X_test)

# Overall metrics are measured in hectograms per hectare.
mae = mean_absolute_error(actual_yield, predicted_yield)
rmse = mean_squared_error(actual_yield, predicted_yield) ** 0.5
r2 = r2_score(actual_yield, predicted_yield)

overall_metrics = {
    "mae_hg_ha": float(mae),
    "rmse_hg_ha": float(rmse),
    "r2": float(r2),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT_DIR / "yield_overall_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(overall_metrics, file, indent=2)

# Save each held-out prediction for later review.
results = X_test.copy()
results["actual_tonnes_ha"] = actual_yield / 10000
results["predicted_tonnes_ha"] = predicted_yield / 10000
results["absolute_error_tonnes_ha"] = np.abs(
    results["actual_tonnes_ha"] - results["predicted_tonnes_ha"]
)
results.to_csv(OUTPUT_DIR / "yield_test_predictions.csv", index=False)

# Summarize the average error for each crop.
crop_metrics = (
    results.groupby("item")
    .agg(
        samples=("item", "size"),
        mean_absolute_error_tonnes_ha=("absolute_error_tonnes_ha", "mean"),
    )
    .reset_index()
    .sort_values("mean_absolute_error_tonnes_ha", ascending=False)
)
crop_metrics.to_csv(OUTPUT_DIR / "yield_crop_metrics.csv", index=False)

print("Overall evaluation:")
print(f"MAE:  {mae:.2f} hg/ha")
print(f"RMSE: {rmse:.2f} hg/ha")
print(f"R²:   {r2:.3f}")

print("\nYield error by crop:")
print(crop_metrics.to_string(index=False))

print("\nSaved overall, per-crop, and detailed results in models/.")