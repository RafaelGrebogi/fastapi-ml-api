import os
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Import TensorFlow handler
# from tensorflow_handler import train_model as tf_train_model, predict_samples as tf_predict_samples

#  MODE SELECTOR 
USE_TENSORFLOW = False  # ➔ Set True to use TensorFlow, False to use RandomForest

MODEL_PATH = Path("models/model.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def run_ml_pipeline(mode: str, data_path: str) -> dict:
    df = pd.read_csv(data_path)

    if mode == "training":
        return train_model(df)
    elif mode == "testing":
        return test_model(df)
    elif mode == "production":
        return predict_model(df)
    else:
        raise ValueError(f"Unknown mode: {mode}")



def train_model(df: pd.DataFrame) -> dict:
    df = df.drop(columns=[col for col in df.columns if col in ("window_id", "msg_id")], errors="ignore")

    X = df.drop(columns=["label"])
    y = df["label"]

    if USE_TENSORFLOW:
        from tensorflow_handler import train_model as tf_train_model
        print(" Training TensorFlow model...")
        y_encoded = encode_labels(y)
        tf_train_model(X.values, y_encoded, epochs=25)
        return {
            "status": "TensorFlow training complete",
            "samples": len(X),
        }
    else:
        print(" Training RandomForest model...")
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        return {
            "status": "RandomForest training complete",
            "model_path": str(MODEL_PATH),
            "samples": len(X),
        }




def test_model(df: pd.DataFrame) -> dict:
    df = df.drop(columns=[col for col in df.columns if col in ("window_id", "msg_id")], errors="ignore")

    X = df.drop(columns=["label"])
    y_true = df["label"]

    if USE_TENSORFLOW:
        from tensorflow_handler import predict_samples as tf_predict_samples
        print(" Testing TensorFlow model...")
        y_encoded = encode_labels(y_true)
        y_pred = tf_predict_samples(X.values)

        report = classification_report(y_encoded, y_pred, output_dict=True)
    else:
        print(" Testing RandomForest model...")
        if not MODEL_PATH.exists():
            return {"error": "Model not found. Please train first."}
        model = joblib.load(MODEL_PATH)
        y_pred = model.predict(X)
        report = classification_report(y_true, y_pred, output_dict=True)

    results_path = RESULTS_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": report,
            "predictions": y_pred.tolist(),
            "targets": y_true.tolist()
        }, f, indent=2)

    return {
        "status": "testing complete",
        "results_file": str(results_path),
        "samples": len(X),
    }





def predict_model(df: pd.DataFrame) -> dict:
    if USE_TENSORFLOW:
        from tensorflow_handler import predict_samples as tf_predict_samples
        print(" Predicting with TensorFlow model...")
        y_pred = tf_predict_samples(df.values)
    else:
        print(" Predicting with RandomForest model...")
        if not MODEL_PATH.exists():
            return {"error": "Model not found. Please train first."}
        model = joblib.load(MODEL_PATH)
        y_pred = model.predict(df)

    results_path = RESULTS_DIR / "predictions.json"
    with open(results_path, "w") as f:
        json.dump({
            "predictions": y_pred.tolist()
        }, f, indent=2)

    return {
        "status": "prediction complete",
        "results_file": str(results_path),
        "samples": len(df),
    }

# --- Helper function for TensorFlow ---
from sklearn.preprocessing import LabelEncoder
_encoder = LabelEncoder()

def encode_labels(y_series):
    return _encoder.fit_transform(y_series)
