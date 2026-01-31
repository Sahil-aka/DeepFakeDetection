# src/model_resnext.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_resnext(input_shape=(160,160,3), classes=1):
    """
    Build a lightweight ResNeXt-like architecture.
    This is NOT the massive 50-layer ImageNet version.
    It is custom-made to run on Low end GPUs
    """

    inputs = layers.Input(shape=input_shape)

    # ---- Stem ----
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # ---- ResNeXt block (grouped conv) ----
    def resnext_block(x, filters, strides=1, cardinality=8):
        # Split input into groups
        groups = []
        for _ in range(cardinality):
            g = layers.Conv2D(filters//cardinality, 3, padding='same', strides=strides, activation='relu')(x)
            groups.append(g)
        x = layers.Concatenate()(groups)
        x = layers.Conv2D(filters, 1, activation='relu')(x)
        return x

    # ---- Stack of ResNeXt blocks ----
    x = resnext_block(x, 64)
    x = layers.MaxPooling2D(2)(x)

    x = resnext_block(x, 128)
    x = layers.MaxPooling2D(2)(x)

    x = resnext_block(x, 256)
    x = layers.MaxPooling2D(2)(x)

    # ---- Classification head ----
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = models.Model(inputs, outputs, name="ResNeXt_Custom")
    return model
