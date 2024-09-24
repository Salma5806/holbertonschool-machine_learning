#!/usr/bin/env python3
"""Preprocess BTC Data"""

import pandas as pd
import numpy as np


def preprocess_data(file_path):
    """Preprocesses The Data"""
    # Load the raw dataset
    data = pd.read_csv(file_path)

    # Drop rows with NaN values
    data = data.dropna()

    # Resample the data to every 60th row
    data = data.iloc[::60, :]

    # Convert 'Timestamp' to datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    data.reset_index(drop=True, inplace=True)

    # Filter data for the year 2017 and onwards
    data = data[data['Timestamp'].dt.year >= 2017]
    data.reset_index(drop=True, inplace=True)

    # Save the preprocessed data
    data.to_csv('preprocessed_data.csv', index=False)

    # Separate into train, validation, and test sets
    n = len(data)
    train_data = data.iloc[:int(n * 0.7)].copy()
    val_data = data.iloc[int(n * 0.7):int(n * 0.9)].copy()
    test_data = data.iloc[int(n * 0.9):].copy()

    # Calculate mean and std from the training set for only numerical columns
    numerical_cols = train_data.select_dtypes(
        include=[np.number]).columns.tolist()

    train_mean = train_data[numerical_cols].mean()

    train_std = train_data[numerical_cols].std()

    # Normalize the training data
    train_data.loc[:, numerical_cols] = (
        train_data[numerical_cols] - train_mean) / train_std

    # Normalize the validation data
    val_data.loc[:, numerical_cols] = (
        val_data[numerical_cols] - train_mean) / train_std

    # Normalize the test data
    test_data.loc[:, numerical_cols] = (
        test_data[numerical_cols] - train_mean) / train_std

    # Save the normalized data
    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

    return train_data, val_data, test_data


if __name__ == "__main__":
    file_path = "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
    preprocess_data(file_path)