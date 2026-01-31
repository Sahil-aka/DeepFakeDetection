# src/train_resnext.py
import tensorflow as tf
from pathlib import Path
from src.data_utils import make_image_datasets
from src.model_resnext import build_resnext

# ---- SAFE SETTINGS (same as CNN) ----
BASE_DIR = "../data/Dataset"
IMG_SIZE = (160, 160)
BATCH = 8
EPOCHS = 5

# Make folders
Path("src/models").mkdir(exist_ok=True)
Path("src/logs/resnext").mkdir(parents=True, exist_ok=True)

MODEL_OUT = "models/resnext_best.h5"

# ---- Load data ----
train_ds, val_ds, test_ds = make_image_datasets(BASE_DIR, IMG_SIZE, BATCH)

# ---- Build ResNeXt ----
model = build_resnext(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---- Callbacks ----
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy"),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
]

# ---- Training ----
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save("models/resnext_final.h5")
print("Saved:", "models/resnext_final.h5")
