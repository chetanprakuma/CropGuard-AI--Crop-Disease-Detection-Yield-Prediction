import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="GropGuard AI",
    page_icon="🌱",
    layout="wide",
)

DISEASE_MODEL_PATH = Path("models/plant_disease_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
DISEASE_REPORT_PATH = Path("models/disease_classification_report.json")

YIELD_MODEL_PATH = Path("models/yield_model.joblib")
YIELD_DATA_PATH = Path("data/yield/df_yield_merged.csv")
YIELD_METRICS_PATH = Path("models/yield_overall_metrics.json")
YIELD_CROP_METRICS_PATH = Path("models/yield_crop_metrics.csv")
IMPORTANCE_PATH = Path("models/yield_feature_importance.csv")


@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(DISEASE_MODEL_PATH)


@st.cache_resource
def load_yield_model():
    return joblib.load(YIELD_MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_yield_data():
    return pd.read_csv(YIELD_DATA_PATH)


st.title("🌱 GropGuard AI")
st.subheader("Crop Disease Detection and Yield Prediction")
st.write(
    "A student project that classifies leaf images and estimates crop yield "
    "using historical country-level data."
)

disease_tab, yield_tab, performance_tab = st.tabs(
    ["🌿 Disease detection", "🌾 Yield estimate", "📊 Model performance"]
)

with disease_tab:
    st.header("Crop disease detection")
    st.write("Upload a leaf photo to see the model's three most likely categories.")

    uploaded_image = st.file_uploader(
        "Choose a leaf photo",
        type=["jpg", "jpeg", "png"],
        key="leaf_image",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded leaf image")

        disease_model = load_disease_model()
        class_names = load_class_names()

        resized_image = image.resize((128, 128))
        image_array = np.array(resized_image, dtype=np.float32)
        image_batch = np.expand_dims(image_array, axis=0)

        probabilities = disease_model.predict(image_batch, verbose=0)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        st.subheader("Top model predictions")

        for rank, index in enumerate(top_indices, start=1):
            label = class_names[index]
            readable_label = label.replace("___", " — ").replace("_", " ")
            confidence = float(probabilities[index])

            st.write(f"{rank}. **{readable_label}** — {confidence * 100:.1f}%")
            st.progress(confidence)

        if float(probabilities[top_indices[0]]) < 0.60:
            st.warning(
                "The model is uncertain about this image. Treat these results "
                "as possibilities, not a confirmed diagnosis."
            )

        st.caption(
            "This model was trained on PlantVillage images. Field photos may "
            "produce less reliable results."
        )

with yield_tab:
    st.header("Historical crop yield estimate")
    st.write(
        "Choose a country, crop, and historical year. The app loads matching "
        "country-year weather and pesticide values as editable starting values."
    )

    yield_data = load_yield_data()

    countries = sorted(yield_data["area"].dropna().unique().tolist())
    crops = sorted(yield_data["item"].dropna().unique().tolist())

    evaluation_years = sorted(
        int(year)
        for year in yield_data["year"].dropna().unique()
        if int(year) >= 2010
    )

    if not evaluation_years:
        st.error("No evaluation years were found in the yield dataset.")
    else:
        country = st.selectbox("Country / area", countries)
        crop = st.selectbox("Crop", crops)
        year = st.selectbox("Historical year", evaluation_years)

        country_year_rows = yield_data[
            (yield_data["area"] == country)
            & (yield_data["year"] == year)
        ]

        def historical_default(column_name):
            if not country_year_rows.empty:
                value = country_year_rows[column_name].median()
                if pd.notna(value):
                    return float(value)

            return float(yield_data[column_name].median())

        rainfall_default = historical_default(
            "average_rain_fall_mm_per_year"
        )
        pesticides_default = historical_default("pesticides_tonnes")
        temperature_default = historical_default("avg_temp")

        with st.form("yield_prediction_form"):
            rainfall = st.number_input(
                "Average annual rainfall (mm)",
                min_value=0.0,
                value=rainfall_default,
                step=10.0,
                key=f"rainfall_{country}_{year}",
            )

            pesticides = st.number_input(
                "Pesticide use (tonnes)",
                min_value=0.0,
                value=pesticides_default,
                step=1.0,
                key=f"pesticides_{country}_{year}",
            )

            temperature = st.number_input(
                "Average temperature (°C)",
                min_value=-5.0,
                max_value=50.0,
                value=temperature_default,
                step=0.5,
                key=f"temperature_{country}_{year}",
            )

            submit_prediction = st.form_submit_button("Estimate yield")

        if submit_prediction:
            input_data = pd.DataFrame(
                [
                    {
                        "area": country,
                        "year": year,
                        "average_rain_fall_mm_per_year": rainfall,
                        "pesticides_tonnes": pesticides,
                        "avg_temp": temperature,
                        "item": crop,
                    }
                ]
            )

            yield_model = load_yield_model()
            predicted_hg_ha = float(yield_model.predict(input_data)[0])
            predicted_tonnes_ha = predicted_hg_ha / 10000

            result_record = {
                "country": country,
                "crop": crop,
                "year": year,
                "rainfall_mm": rainfall,
                "pesticides_tonnes": pesticides,
                "average_temperature_c": temperature,
                "predicted_yield_tonnes_per_hectare": predicted_tonnes_ha,
                "recorded_yield_tonnes_per_hectare": None,
                "absolute_error_tonnes_per_hectare": None,
            }

            st.subheader("Model estimate")
            st.metric(
                "Estimated yield",
                f"{predicted_tonnes_ha:,.2f} tonnes per hectare",
            )

            actual_rows = yield_data[
                (yield_data["area"] == country)
                & (yield_data["item"] == crop)
                & (yield_data["year"] == year)
            ]

            if not actual_rows.empty:
                actual_hg_ha = float(actual_rows["value_hg_ha"].median())
                actual_tonnes_ha = actual_hg_ha / 10000
                absolute_error = abs(predicted_tonnes_ha - actual_tonnes_ha)

                result_record["recorded_yield_tonnes_per_hectare"] = (
                    actual_tonnes_ha
                )
                result_record["absolute_error_tonnes_per_hectare"] = (
                    absolute_error
                )

                st.write(
                    f"**Recorded yield:** {actual_tonnes_ha:,.2f} tonnes/ha"
                )
                st.write(
                    f"**Absolute difference:** {absolute_error:,.2f} tonnes/ha"
                )
            else:
                st.info(
                    "The dataset has no recorded yield for this exact "
                    "country, crop, and year combination."
                )

            st.caption(
                "This is a historical dataset comparison, not a current-season "
                "forecast for an individual farm."
            )

            csv_data = pd.DataFrame([result_record]).to_csv(index=False)

            st.download_button(
                label="Download this result as CSV",
                data=csv_data,
                file_name="gropguard_yield_estimate.csv",
                mime="text/csv",
            )

    if IMPORTANCE_PATH.exists():
        importance_data = pd.read_csv(IMPORTANCE_PATH)

        st.subheader("Yield model input importance")
        st.bar_chart(
            importance_data.set_index("input_group")["importance_percent"]
        )
        st.caption(
            "These scores show relative model importance; they do not show "
            "that an input causes a change in yield."
        )

with performance_tab:
    st.header("Model performance")
    st.write(
        "These scores summarize performance on the held-out validation or "
        "historical evaluation data."
    )

    if DISEASE_REPORT_PATH.exists():
        with open(DISEASE_REPORT_PATH, "r", encoding="utf-8") as file:
            disease_report = json.load(file)

        st.subheader("Disease classifier")

        disease_accuracy = disease_report.get("accuracy")
        macro_scores = disease_report.get("macro avg", {})
        weighted_scores = disease_report.get("weighted avg", {})

        metric_columns = st.columns(3)

        if disease_accuracy is not None:
            metric_columns[0].metric(
                "Validation accuracy",
                f"{disease_accuracy:.1%}",
            )

        if "f1-score" in macro_scores:
            metric_columns[1].metric(
                "Macro F1",
                f"{macro_scores['f1-score']:.3f}",
            )

        if "f1-score" in weighted_scores:
            metric_columns[2].metric(
                "Weighted F1",
                f"{weighted_scores['f1-score']:.3f}",
            )

        class_rows = []

        for class_name, scores in disease_report.items():
            if (
                isinstance(scores, dict)
                and "f1-score" in scores
                and class_name not in {"macro avg", "weighted avg", "micro avg"}
            ):
                class_rows.append(
                    {
                        "class": class_name.replace("___", " — ").replace(
                            "_", " "
                        ),
                        "f1_score": scores["f1-score"],
                    }
                )

        if class_rows:
            lowest_f1 = (
                pd.DataFrame(class_rows)
                .sort_values("f1_score")
                .head(10)
                .set_index("class")
            )

            st.write("Lowest 10 category F1 scores")
            st.bar_chart(lowest_f1["f1_score"])
    else:
        st.info(
            "Disease evaluation files are missing. Run "
            "`python evaluate_disease.py` in Terminal."
        )

    st.subheader("Yield model")

    if YIELD_METRICS_PATH.exists():
        with open(YIELD_METRICS_PATH, "r", encoding="utf-8") as file:
            yield_metrics = json.load(file)

        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "MAE",
            f"{yield_metrics['mae_hg_ha'] / 10000:.2f} tonnes/ha",
        )
        metric_columns[1].metric(
            "RMSE",
            f"{yield_metrics['rmse_hg_ha'] / 10000:.2f} tonnes/ha",
        )
        metric_columns[2].metric(
            "R²",
            f"{yield_metrics['r2']:.3f}",
        )
    else:
        st.info(
            "Yield evaluation files are missing. Run "
            "`python evaluate_yield.py` in Terminal."
        )

    if YIELD_CROP_METRICS_PATH.exists():
        crop_metrics = pd.read_csv(YIELD_CROP_METRICS_PATH)
        crop_chart = crop_metrics.set_index("item")[
            "mean_absolute_error_tonnes_ha"
        ]

        st.write("Mean absolute error by crop")
        st.bar_chart(crop_chart)