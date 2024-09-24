#!/usr/bin/env python3
"""Forecast Bitcoin"""

import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import load_model, Sequential
import pandas as pd
import matplotlib.pyplot as plt

# Load preprocessed data
train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')
test_data = pd.read_csv('test_data.csv')

# Make sure # of rows in train dataset is a multiple of 24
num_rows_train = len(train_data)
num_rows_train = num_rows_train - num_rows_train % 24
train_data = train_data.head(num_rows_train)

# Make sure # of rows in validation dataset is a multiple of 24
num_rows_val = len(val_data)
num_rows_val = num_rows_val - num_rows_val % 24
val_data = val_data.head(num_rows_val)

# Make sure # of rows in test dataset is a multiple of 24
num_rows_test = len(test_data)
num_rows_test = num_rows_test - num_rows_test % 24
test_data = test_data.head(num_rows_test)

# Reshape Features for LSTM
train_features = tf.reshape(train_data['Close'].values, (-1, 24, 1))
val_features = tf.reshape(val_data['Close'].values, (-1, 24, 1))
test_features = tf.reshape(test_data['Close'].values, (-1, 24, 1))

# Shift 'Close' by 1 to create labels
train_labels = train_data['Close'].shift(1).fillna(0).values.reshape(-1, 24, 1)
val_labels = val_data['Close'].shift(1).fillna(0).values.reshape(-1, 24, 1)
test_labels = test_data['Close'].shift(1).fillna(0).values.reshape(-1, 24, 1)

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    train_features, train_labels))

val_dataset = tf.data.Dataset.from_tensor_slices((
    val_features, val_labels))

# Build LSTM model
model = Sequential([
    LSTM(24, input_shape=(24, 1), return_sequences=True),
    LSTM(24, return_sequences=True),
    Dense(1)
])

# Compile model with Adam optimizer and MSE loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train model
history = model.fit(train_dataset.batch(32),
                    epochs=20,
                    validation_data=val_dataset.batch(32))

# Use validation data to evaluate model
loss = model.evaluate(val_features, val_labels)
print(f'Mean Squared Error on Validation Data: {loss}')

# Plot Training & Validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save Plot as PNG image
plt.savefig('loss_plot.png')

# Save model
model.save('btc_forecast_model.tf')


# Define prediction model
def predict(model, input_data):
    """Prediction for BTC"""
    predictions = model.predict(input_data)
    return predictions


# Load saved model
model = load_model('btc_forecast_model.tf')

# Use model to make a prediction
input_data = test_features
predictions = predict(model, input_data)

print(predictions)

# Reshape predictions and test_labels for plotting
predictions = predictions.reshape(-1)
test_labels = test_labels.reshape(-1)

# Create a figure
plt.figure(figsize=(10, 5))

# Plot actual values
plt.plot(test_labels, label='Actual')

# Plot predicted values
plt.plot(predictions, label='Predicted')

# Set title
plt.title('BTC Forecast vs Actual')

# Show the legend
plt.legend()

# Display the plot
plt.show()