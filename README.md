# GropGuard AI

A final-semester project for crop leaf disease classification and historical
crop yield estimation, built with Python, TensorFlow, scikit-learn, and Streamlit.

## Features

- Classifies uploaded leaf images into PlantVillage crop and disease categories.
- Estimates historical yield per hectare from country, crop, year, rainfall,
  pesticide use, and temperature.
- Displays disease results and yield estimates in a Streamlit web app.

## Project files

- `app.py` — Streamlit user interface
- `train_disease.py` — trains the leaf image classifier
- `train_yield.py` — trains and evaluates the yield model
- `requirements.txt` — required Python packages
- `data/plantvillage/raw/color/` — color leaf images
- `data/yield/df_yield_merged.csv` — yield dataset
- `models/` — saved models and disease class names

## Run on macOS

Open Terminal in the project folder and run:

```bash
source project/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Train the models

With the `project` environment active, run:

```bash
python train_disease.py
python train_yield.py
```

## Data sources

- Disease images: [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- Yield data: [Crop Yield Prediction dataset on Zenodo](https://zenodo.org/records/18761663)

## Latest results

- Disease classifier: **93.05% validation accuracy** on the PlantVillage image split.
- Yield model: **MAE 10,614.53 hg/ha**, **RMSE 20,811.34 hg/ha**, and **R² 0.952** on later historical years.

## Limitations

The disease classifier was trained on PlantVillage images, so results may be less
reliable on field photos. The yield dataset contains historical country-level
records; its estimates are for demonstration and are not farm-specific forecasts.

### Disease image check

- Dataset label: Tomato Early blight
- Model's top prediction: Tomato Early blight
- Confidence: 94.1%