import os
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from supabase_client import supabase
from utils.service_utils import get_service_details, upload_result_to_db
from context_vars import current_user_id, current_service_id, current_DeviceId

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
        return train_model(df, data_path)
    elif mode == "testing":
        return test_model(df)
    elif mode == "production":
        return predict_model(df)
    else:
        raise ValueError(f"Unknown mode: {mode}")



def train_model(df: pd.DataFrame, data_path: str) -> dict:
    from datetime import datetime

    df = df.drop(columns=[col for col in df.columns if col in ("window_id", "msg_id")], errors="ignore")
    X = df.drop(columns=["label"])
    y = df["label"]

    # Get timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Archive directory structure
    ARCHIVE_DIR = Path("model_archive") / timestamp
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Metadata preparation
    metadata = {
        "timestamp": timestamp,
        "training_file": data_path,
        "samples": len(X)
    }

    ServiceId = current_service_id.get()
    success, service_details = get_service_details(ServiceId, supabase)
    ml_method = service_details.get("ml_method")
    DeviceId = service_details.get("device_id")
    current_DeviceId.set(DeviceId)

    if USE_TENSORFLOW and ml_method and ml_method.get("id") == 2:
        from tensorflow_handler import train_model as tf_train_model
        print(" Training TensorFlow model...")
        y_encoded = encode_labels(y)

        # Train TensorFlow model
        model = tf_train_model(X.values, y_encoded, epochs=25)

        # Save the active model
        active_model_path = Path("models/model.keras")
        model.save(active_model_path)

        # Save the archived model with timestamp
        archive_model_path = ARCHIVE_DIR / f"model_{timestamp}.keras"
        model.save(archive_model_path)

        # Update metadata
        metadata.update({
            "method": "TensorFlow",
            "model_path": str(archive_model_path)
        })

        # Save metadata
        metadata_path = ARCHIVE_DIR / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload metadata to the database
        upload_response = upload_result_to_db(json_data=metadata, supabase=supabase, isDev=True)

        print(f" TensorFlow model archive created at: {ARCHIVE_DIR}")

        return {
            "status": "TensorFlow training complete",
            "model_path": str(active_model_path),
            "archive_path": str(archive_model_path),
            "samples": len(X),
        }

    elif ml_method and ml_method.get("id") == 1:
        print(" Training RandomForest model...")
        model = RandomForestClassifier()
        model.fit(X, y)

        # Save the active model
        joblib.dump(model, MODEL_PATH)

        # Save the archived model with timestamp
        archive_model_path = ARCHIVE_DIR / f"model_{timestamp}.pkl"
        joblib.dump(model, archive_model_path)

        # Update metadata
        metadata.update({
            "method": "RandomForest",
            "model_path": str(archive_model_path)
        })

        # Save metadata
        metadata_path = ARCHIVE_DIR / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload metadata to the database
        upload_response = upload_result_to_db(json_data=metadata, supabase=supabase, isDev=True)

        # Optional: check if it succeeded
        if upload_response["success"]:
            print("Result metadata uploaded successfully.")
        else:
            print("Upload failed:", upload_response["message"])

        print(f" RandomForest model archive created at: {ARCHIVE_DIR}")

        return {
            "status": "RandomForest training complete",
            "model_path": str(MODEL_PATH),
            "archive_path": str(archive_model_path),
            "samples": len(X),
        }






def test_model(df: pd.DataFrame) -> dict:
    df = df.drop(columns=[col for col in df.columns if col in ("window_id", "msg_id")], errors="ignore")

    X = df.drop(columns=["label"])
    y_true = df["label"]

    ServiceId = current_service_id.get()
    success, service_details = get_service_details(ServiceId, supabase)
    ml_method = service_details.get("ml_method")

    if USE_TENSORFLOW and ml_method and ml_method.get("id") == 2:
        from tensorflow_handler import predict_samples as tf_predict_samples
        print(" Testing TensorFlow model...")
        y_encoded = encode_labels(y_true)
        y_pred = tf_predict_samples(X.values)

        report = classification_report(y_encoded, y_pred, output_dict=True)
        # Calculate correct predictions
        corrects = (y_encoded == y_pred).tolist()

    elif ml_method and ml_method.get("id") == 1:
        print(" Testing RandomForest model...")
        if not MODEL_PATH.exists():
            return {"error": "Model not found. Please train first."}
        model = joblib.load(MODEL_PATH)
        y_pred = model.predict(X)
        report = classification_report(y_true, y_pred, output_dict=True)
        # Calculate correct predictions
        corrects = (y_pred == y_true).tolist()

    results_path = RESULTS_DIR / "test_results.json"

    # Prepare result data
    result_data = {
        "metrics": report,
        "predictions": y_pred.tolist(),
        "targets": y_true.tolist(),
        "corrects": corrects
    }
    with open(results_path, "w") as f:
        json.dump(result_data, f, indent=2)
    # with open(results_path, "w") as f:
    #     json.dump({
    #         "metrics": report,
    #         "predictions": y_pred.tolist(),
    #         "targets": y_true.tolist(),
    #         "corrects": corrects
    #     }, f, indent=2)


    # Upload test_results to the database
    upload_response = upload_result_to_db(json_data=result_data, supabase=supabase, isDev=True)

    # Optional: check if it succeeded
    if upload_response["success"]:
        print("Result metadata uploaded successfully.")
    else:
        print("Upload failed:", upload_response["message"])



    print(" Testing completed!")
    return {
        "status": "testing complete",
        "results_file": str(results_path),
        "samples": len(X),
    }





def predict_model(df: pd.DataFrame) -> dict:
    df = df.drop(columns=[col for col in df.columns if col in ("window_id", "msg_id")], errors="ignore")

    X = df.drop(columns=["label"])

    ServiceId = current_service_id.get()
    success, service_details = get_service_details(ServiceId, supabase)
    ml_method = service_details.get("ml_method")

    if USE_TENSORFLOW and ml_method and ml_method.get("id") == 2:
        from tensorflow_handler import predict_samples as tf_predict_samples
        print(" Predicting with TensorFlow model...")
        y_pred = tf_predict_samples(X.values)

    elif ml_method and ml_method.get("id") == 1:
        print(" Predicting with RandomForest model...")
        if not MODEL_PATH.exists():
            return {"error": "Model not found. Please train first."}
        model = joblib.load(MODEL_PATH)
        y_pred = model.predict(X)

    results_path = RESULTS_DIR / "predictions.json"
    # with open(results_path, "w") as f:
    #     json.dump({
    #         "predictions": y_pred.tolist()
    #     }, f, indent=2)

    # Prepare result data
    result_data = {
        "predictions": y_pred.tolist()
    }
    with open(results_path, "w") as f:
        json.dump(result_data, f, indent=2)

    # Upload test_results to the database
    upload_response = upload_result_to_db(json_data=result_data, supabase=supabase, isDev=False)

    # Optional: check if it succeeded
    if upload_response["success"]:
        print("Result metadata uploaded successfully.")
    else:
        print("Upload failed:", upload_response["message"])

    print(" Prediction completed!")
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
