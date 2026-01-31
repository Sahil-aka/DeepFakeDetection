# src/model_lstm.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm(input_shape=(160,160,3)):
    """
    LSTM-based model for deepfake detection.
    Uses CNN for feature extraction, then LSTM to process spatial features as sequences.
    """
    inputs = layers.Input(shape=input_shape)
    
    # ---- CNN Feature Extractor ----
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # ---- Reshape for LSTM ----
    # After 3 pooling layers: 160->80->40->20
    # Shape: (batch, 20, 20, 128)
    # Reshape to treat spatial dimensions as sequence
    # We'll treat each row as a time step
    shape = tf.keras.backend.int_shape(x)
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)  # (batch, 20, 20*128) = (batch, 20, 2560)
    
    # ---- LSTM Layers ----
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)  # Final LSTM doesn't return sequences
    x = layers.Dropout(0.3)(x)
    
    # ---- Classification Head ----
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_DeepFake")
    return model
