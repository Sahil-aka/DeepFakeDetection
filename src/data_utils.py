# src/data_utils.py
import tensorflow as tf
from pathlib import Path

AUTOTUNE = tf.data.AUTOTUNE

def make_image_datasets(base_dir="../data/Dataset", img_size=(224,224), batch=32):
    base = Path(base_dir)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base/"Train", image_size=img_size, batch_size=batch, shuffle=True
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base/"Validation", image_size=img_size, batch_size=batch, shuffle=False
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base/"Test", image_size=img_size, batch_size=batch, shuffle=False
    )

    # simple normalization & performance
    def norm(x,y):
        x = tf.cast(x, tf.float32) / 255.0
        return x,y

    train_ds = train_ds.map(norm, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.map(norm,   num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.map(norm,  num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds
