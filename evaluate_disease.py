import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = Path("data/plantvillage/raw/color")
MODEL_PATH = Path("models/plant_disease_model.keras")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the same seeded, shuffled split as the training script.
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    shuffle=True,
)

class_names = validation_data.class_names
model = tf.keras.models.load_model(MODEL_PATH)

# Collect each batch's labels and predictions together so they stay aligned.
true_labels = []
predicted_labels = []

for images, labels in validation_data:
    probabilities = model.predict(images, verbose=0)
    batch_predictions = np.argmax(probabilities, axis=1)

    true_labels.extend(labels.numpy().tolist())
    predicted_labels.extend(batch_predictions.tolist())

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
all_class_numbers = list(range(len(class_names)))

report_text = classification_report(
    true_labels,
    predicted_labels,
    labels=all_class_numbers,
    target_names=class_names,
    digits=3,
    zero_division=0,
)

report_data = classification_report(
    true_labels,
    predicted_labels,
    labels=all_class_numbers,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

print("\nDisease model evaluation:")
print(report_text)

with open(
    OUTPUT_DIR / "disease_classification_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(report_data, file, indent=2)

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=all_class_numbers,
)

matrix_table = pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names,
)
matrix_table.to_csv(OUTPUT_DIR / "disease_confusion_matrix.csv")

print("Saved the classification report and confusion matrix in models/.")