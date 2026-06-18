reates, trains, and validates an RNN model for forecasting Bitcoin prices
using an input pipeline built with tf.data.Dataset.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def create_dataset(data, window_size=24):
    """
    Uses tf.data.Dataset to create a sliding window pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    # Create windows of size 25 (24 inputs + 1 target)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # Split into features (past 24h) and target (next 1h 'Weighted_Price')
    # Assuming Weighted_Price is at index -1
    dataset = dataset.map(lambda window: (window[:-1], window[-1, -1]))
    return dataset


def build_model(input_shape):
    """
    Builds a Keras RNN model using GRU/LSTM layers.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == "__main__":
    # Load the preprocessed hourly dataset
    df = pd.read_csv('cleaned_coinbase_hourly.csv', index_index=0)
    
    # Feature Selection: Scale data between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    # Train/Validation Split (80/20)
    split_idx = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_idx]
    val_data = scaled_data[split_idx:]
    
    # Create tf.data.Dataset pipelines
    BATCH_SIZE = 64
    train_dataset = create_dataset(train_data).shuffle(10000).batch(BATCH_SIZE).prefetch(1)
    val_dataset = create_dataset(val_data).batch(BATCH_SIZE).prefetch(1)
    
    # Build and Train Model
    # Input shape: (time_steps=24, features=num_columns)
    model = build_model((24, scaled_data.shape[1]))
    
    print(model.summary())
    
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset
    )
    
    # Save the trained model
    model.save('btc_forecast_model.h5')
