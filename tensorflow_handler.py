import os
import json
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np

MODEL_DIR = "tf_models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "walking_model.h5")

RESULTS_DIR = "tf_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

encoder = LabelEncoder()

def build_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(X, y, epochs=10):
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    if os.path.exists(MODEL_PATH):
        print("📦 Loading existing model...")
        model = keras.models.load_model(MODEL_PATH)
    else:
        print("🛠 Building new model...")
        model = build_model(input_dim, num_classes)

    history = model.fit(X, y, epochs=epochs, batch_size=32)

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")

    # ✅ Save training history
    save_training_history(history.history)

    return model

def predict_samples(X):
    model = keras.models.load_model(MODEL_PATH)
    predictions = model.predict(X)
    predicted_classes = predictions.argmax(axis=1)
    return predicted_classes

def save_training_history(history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(RESULTS_DIR, f"training_history_{timestamp}.json")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"📈 Training history saved at: {history_path}")
