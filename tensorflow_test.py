import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from keras import layers


# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),  # First hidden layer
    layers.Dense(64, activation="relu"),  # Second hidden layer
    layers.Dense(1)  # Output layer for regression (single value)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # Mean Squared Error for regression

model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1)

y_pred = model.predict(X_test)

# Print first 5 predictions
print("First 5 Predictions:", y_pred[:5].flatten())



y_pred = model.predict(X_test)  # Get predictions
mse = mean_squared_error(y_test, y_pred)  # Compute MSE

print(f"Mean Squared Error (MSE): {mse:.2f}")