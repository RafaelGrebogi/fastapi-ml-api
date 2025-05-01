import firebase_admin
from firebase_admin import credentials, db
import os
import json
from datetime import datetime
import asyncio
from feature_extraction import extract_features_from_firebase_batch
from ml_manager import run_ml_pipeline

# Firebase setup
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://esp32-datalogger-c9c32-default-rtdb.asia-southeast1.firebasedatabase.app/"  # Firebase URL
    })

# Paths
CONTROL_PATH = "/ControlFlag/"
DATA_PATH = "/ESP32_Develop/TrainingDataset/"
ARCHIVE_PATH = "/ESP32_Develop/TrainingArchive/"
TRAINING_DATA_DIR = "data/training/"

TESTING_DATA_DIR = "data/testing/"
TESTING_CSV_PATH = "data/testing/testing_features.csv"
FIREBASE_TESTING_PATH = "/ESP32_Develop/TestingDataset/"

PRODUCTION_DATA_DIR = "data/production/"
PRODUCTION_CSV_PATH = "data/production/production_features.csv"
FIREBASE_PRODUCTION_PATH = "/ESP32_Develop/Data/"

# Ensure folder exists
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(TESTING_DATA_DIR, exist_ok=True)
os.makedirs(PRODUCTION_DATA_DIR, exist_ok=True)

# Main handler to be triggered by FastAPI

# ===========================================
# ===========================================
async def process_training_data():
    try:

        data, data_ref = download_data_if_complete(DATA_PATH, CONTROL_PATH, TRAINING_DATA_DIR, ARCHIVE_PATH)


        csv_path = extract_features_from_firebase_batch(data, USE_MULTI_MESSAGE_WINDOW=True)

        # Run training ML pipeline
        ml_result = run_ml_pipeline("training", csv_path)
        print("✅ ML Pipeline Result:", ml_result)

        # Clean up original data and control flag
        data_ref.delete()
        # control_ref.set({"complete": False})
        # print("🧹 Training data and control flag cleared.\n")

        return True

    except Exception as e:
        print(f"❌ Exception during processing: {e}")
        return False



# ===========================================
# ===========================================
def process_testing_data():
    print(" Triggered: Testing Mode")

    # Download JSON from Firebase (same logic as training)
    data, _ = download_data_if_complete(FIREBASE_TESTING_PATH, CONTROL_PATH, TESTING_DATA_DIR, ARCHIVE_PATH)

    # Extract features for ML method
    # Extract features from each batch

    csv_path = extract_features_from_firebase_batch(data, USE_MULTI_MESSAGE_WINDOW=True)

    # Run testing pipeline
    result = run_ml_pipeline("testing", csv_path)

    return result

# ===========================================
# ===========================================
def process_production_data():
    print(" Triggered: Production Mode")

    # Download JSON from Firebase
    data, _ = download_data_if_complete(FIREBASE_PRODUCTION_PATH, CONTROL_PATH, PRODUCTION_DATA_DIR, ARCHIVE_PATH)

    # Extract features for ML method
    # Extract features from each batch

    csv_path = extract_features_from_firebase_batch(data, USE_MULTI_MESSAGE_WINDOW=True)

    # Run prediction pipeline
    result = run_ml_pipeline("production", csv_path)

    return result

# ===========================================
# ===========================================


def download_data_if_complete(firebase_data_path: str, firebase_control_path: str, local_dir: str, archive_dir: str) -> str:
    """
    Checks the 'complete' flag in Firebase, downloads the dataset if ready,
    saves it locally, archives it, and deletes it from the original path.
    """
    import time
    from datetime import datetime
    from pathlib import Path

    # Delay to ensure ESP32 has finished uploading
    time.sleep(2)

    control_ref = db.reference(firebase_control_path)
    control_data = control_ref.get()

    if not control_data or control_data.get("complete") != True:
        print("❌ Trigger received, but 'complete' flag not set. Aborting.")
        raise RuntimeError("Control flag not set to 'complete'.")

    # Download data
    data_ref = db.reference(firebase_data_path)
    data = data_ref.get()

    if not data:
        print("❌ No data found at Firebase path.")
        raise RuntimeError("No data found in Firebase.")

    # Save locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_{timestamp}.json"
    save_path = Path(local_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ Archive in Firebase
    archive_ref = db.reference(f"{archive_dir}/{timestamp}")
    archive_ref.set(data)
    print(" Data archived in Firebase.")


    #  Delete original training data from Firebase

    data_ref.delete()
    print(" Original data deleted from Firebase.")

    # Save locally to file
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f" Firebase data saved to: {save_path}")
    return data, archive_ref
