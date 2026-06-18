#!/usr/bin/env python3
"""
Preprocesses raw Coinbase and Bitstamp Bitcoin datasets for time series.
Resamples 1-minute intervals to 1-hour intervals and handles missing values.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_dataset(file_path):
    """
    Loads, cleans, resamples, and scales a raw BTC time-series CSV file.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Convert Unix timestamp to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # Handle missing values during the raw minute phase
    # Volume can be 0 if no trades happened; prices can be forward-filled
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)
    df.ffill(inplace=True)

    # Resample from 1-minute to 1-hour intervals
    # Aggregation mapping to keep data accurate
    resample_logic = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    }
    df_hourly = df.resample('1H').agg(resample_logic)

    # Drop any remaining NaNs that couldn't be forward-filled at the start
    df_hourly.dropna(inplace=True)

    return df_hourly


if __name__ == "__main__":
    # Example execution (Adjust file paths based on local environment)
    print("Preprocessing Coinbase data...")
    coinbase = preprocess_dataset('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
    
    # Save the cleaned dataset to save RAM/compute during model training
    coinbase.to_csv('cleaned_coinbase_hourly.csv')
    print("Saved preprocessed data to cleaned_coinbase_hourly.csv")
