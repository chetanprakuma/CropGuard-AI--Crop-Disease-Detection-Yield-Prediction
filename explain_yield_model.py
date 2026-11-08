from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/yield_model.joblib")
OUTPUT_PATH = Path("models/yield_feature_importance.csv")

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["prepare_data"]
forest = pipeline.named_steps["random_forest"]

encoded_feature_names = preprocessor.get_feature_names_out()
importance_values = forest.feature_importances_

details = pd.DataFrame(
    {
        "encoded_feature": encoded_feature_names,
        "importance": importance_values,
    }
)

def get_input_group(encoded_name):
    if encoded_name.startswith("categories__area_"):
        return "Country / area"

    if encoded_name.startswith("categories__item_"):
        return "Crop"

    if encoded_name.startswith("numbers__"):
        feature_name = encoded_name.replace("numbers__", "", 1)

        readable_names = {
            "year": "Year",
            "average_rain_fall_mm_per_year": "Rainfall",
            "pesticides_tonnes": "Pesticide use",
            "avg_temp": "Temperature",
        }

        return readable_names.get(feature_name, feature_name)

    return encoded_name

details["input_group"] = details["encoded_feature"].apply(get_input_group)

grouped = (
    details.groupby("input_group", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)

grouped["importance_percent"] = grouped["importance"] * 100
grouped.to_csv(OUTPUT_PATH, index=False)

print("Yield model importance by input:")
print(grouped[["input_group", "importance_percent"]].to_string(index=False))
print(f"\nSaved detailed results to {OUTPUT_PATH}")