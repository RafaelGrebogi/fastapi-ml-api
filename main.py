from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib

#from ml_model import model_diabetes, model_calihousing  # Import trained models

# Load pre-trained models
model_diabetes = joblib.load("model_diabetes.pkl")
model_calihousing = joblib.load("model_calihousing.pkl")

app = FastAPI()

@app.post("/predict_diabetes")
def predict(features: dict):
    """
    Receives input features as JSON, processes them, 
    and returns a diabetes prediction using the trained model.
    """
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([features])

    # Make a prediction
    prediction = model_diabetes.predict(df)

    # Return the result as JSON
    return {"prediction": prediction.tolist()}

@app.post("/predict_calihousing")
def predict_calihousing(features: dict):
    """
    Receives input features as JSON, processes them, 
    and returns a prediction using the California Housing model.
    """
    df = pd.DataFrame([features])  # Convert input to DataFrame
    prediction = model_calihousing.predict(df)  # Make prediction
    return {"prediction": prediction.tolist()}  # Return as JSON

