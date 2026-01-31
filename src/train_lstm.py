# src/train_lstm.py
"""
Training script for LSTM model on a memory-constrained laptop (GTX 1650).
- Enables GPU memory growth
- Sets an optional per-process GPU memory limit (MB)
- Uses smaller IMG_SIZE and BATCH by default
- Uses conservative tf.data settings
"""

import os
import tensorflow as tf
from pathlib import Path

# ---------- GPU safety / memory settings ----------
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Try to limit to ~3500 MB for GTX1650
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
            )
            print("Set GPU logical device memory limit: 3500 MB")
        except Exception as e:
            print("Could not set GPU memory limit (continuing):", e)
    else:
        print("No GPU detected by TensorFlow.")
except Exception as e:
    print("GPU configuration failed:", e)

# ---------- Conservative data pipeline settings ----------
DATA_PARALLEL_CALLS = 4
PREFETCH = 1

# ---------- Config ----------
BASE_DIR = "../data/Dataset"
IMG_SIZE = (160, 160)
BATCH = 8
EPOCHS = 5
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = MODEL_DIR / "lstm_best.h5"
LOG_DIR = Path("logs/lstm")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Import model ----------
from model_lstm import build_lstm

# ---------- Data pipeline ----------
AUTOTUNE = tf.data.AUTOTUNE

def make_image_datasets(base_dir=BASE_DIR, img_size=IMG_SIZE, batch=BATCH):
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base.resolve()}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base / "Train",
        image_size=img_size,
        batch_size=batch,
        shuffle=True
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base / "Validation",
        image_size=img_size,
        batch_size=batch,
        shuffle=False
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base / "Test",
        image_size=img_size,
        batch_size=batch,
        shuffle=False
    )

    # Simple normalization function
    def _norm(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    # Apply mapping with conservative parallelism and prefetch=1
    train_ds = train_ds.map(_norm, num_parallel_calls=DATA_PARALLEL_CALLS)
    train_ds = train_ds.prefetch(PREFETCH)

    val_ds = val_ds.map(_norm, num_parallel_calls=DATA_PARALLEL_CALLS)
    val_ds = val_ds.prefetch(PREFETCH)

    test_ds = test_ds.map(_norm, num_parallel_calls=DATA_PARALLEL_CALLS)
    test_ds = test_ds.prefetch(PREFETCH)

    return train_ds, val_ds, test_ds

# ---------- Train ----------
def main():
    print("Starting LSTM training with settings:")
    print(f"  BASE_DIR = {BASE_DIR}")
    print(f"  IMG_SIZE = {IMG_SIZE}, BATCH = {BATCH}, EPOCHS = {EPOCHS}")
    print(f"  DATA_PARALLEL_CALLS = {DATA_PARALLEL_CALLS}, PREFETCH = {PREFETCH}")

    train_ds, val_ds, test_ds = make_image_datasets()

    model = build_lstm(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    final_path = MODEL_DIR / "lstm_final.h5"
    model.save(str(final_path))
    print("Training finished. Best model:", MODEL_OUT)
    print("Final model saved:", final_path)

if __name__ == "__main__":
    main()
