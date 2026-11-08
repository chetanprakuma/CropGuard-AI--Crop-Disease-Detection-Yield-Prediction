from pathlib import Path
import shutil

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/yield/df_yield_merged.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "yield_model.joblib"
BACKUP_PATH = MODEL_DIR / "yield_model_before_deduplication.joblib"

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

original_row_count = len(data)
repeated_rows = int(data.duplicated(subset=key_columns).sum())

print(f"Rows before grouping: {original_row_count}")
print(f"Repeated rows to combine: {repeated_rows}")

# Keep one row per country, year, and crop.
# Median handles repeated climate values while preserving the typical value.
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

print(f"Rows after grouping: {len(data)}")

# Train on earlier years and evaluate on later historical years.
train_data = data[data["year"] <= 2009]
test_data = data[data["year"] >= 2010]

X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_test = test_data[feature_columns]
y_test = test_data[target_column]

categorical_features = ["area", "item"]
numeric_features = [
    "year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            Pipeline(
                steps=[
                    ("fill_missing", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
        (
            "numbers",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
    ]
)

yield_model = Pipeline(
    steps=[
        ("prepare_data", preprocessor),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print(f"Training rows: {len(X_train)}")
print(f"Evaluation rows: {len(X_test)}")
print("Training the yield model...")

yield_model.fit(X_train, y_train)
predictions = yield_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nEvaluation on years 2010 and later:")
print(f"MAE:  {mae:.2f} hg/ha")
print(f"RMSE: {rmse:.2f} hg/ha")
print(f"R²:   {r2:.3f}")

if MODEL_PATH.exists() and not BACKUP_PATH.exists():
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    print(f"Previous model backed up to {BACKUP_PATH}")

joblib.dump(yield_model, MODEL_PATH)
print(f"\nYield model saved to {MODEL_PATH}")