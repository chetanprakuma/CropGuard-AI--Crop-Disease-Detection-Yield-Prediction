import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DATA_DIR = Path("data/plantvillage/raw/color")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
BACKUP_PATH = MODEL_DIR / "plant_disease_model_before_class_weights.keras"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

train_data = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

validation_data = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_data.class_names
number_of_classes = len(class_names)

print(f"Found {number_of_classes} classes.")

# Count the training images in each class.
class_counts = np.zeros(number_of_classes, dtype=np.int64)

for images, labels in train_data:
    class_counts += np.bincount(
        labels.numpy(),
        minlength=number_of_classes,
    )

# Give more training weight to classes with fewer images.
total_images = int(class_counts.sum())
class_weights = {
    class_number: total_images / (number_of_classes * int(class_count))
    for class_number, class_count in enumerate(class_counts)
    if class_count > 0
}

print("Calculated class weights.")
print(
    "Weight range:",
    f"{min(class_weights.values()):.2f} to "
    f"{max(class_weights.values()):.2f}",
)

autotune = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=autotune)
validation_data = validation_data.prefetch(buffer_size=autotune)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

base_model = keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
images = data_augmentation(inputs)
images = keras.applications.mobilenet_v2.preprocess_input(images)
features = base_model(images, training=False)
features = layers.GlobalAveragePooling2D()(features)
features = layers.Dropout(0.25)(features)
outputs = layers.Dense(number_of_classes, activation="softmax")(features)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

print("Training with class weights...")

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    class_weight=class_weights,
)

# Keep a copy of the current model before replacing it.
if MODEL_PATH.exists() and not BACKUP_PATH.exists():
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    print(f"Previous model backed up to {BACKUP_PATH}")

model.save(MODEL_PATH)

with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

print("Training finished.")
print(f"Updated model saved to {MODEL_PATH}")
print("Class names saved to models/class_names.json")