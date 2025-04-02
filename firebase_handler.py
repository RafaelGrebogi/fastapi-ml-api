import firebase_admin
from firebase_admin import credentials, db
import os
import json
from datetime import datetime
import asyncio
from feature_extraction import extract_features_from_firebase_batch

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

# Ensure folder exists
os.makedirs("data/training", exist_ok=True)

# Main handler to be triggered by FastAPI
async def process_training_data():
    try:
        # Small precaution delay in case ESP32 is still writing
        await asyncio.sleep(2)

        control_ref = db.reference(CONTROL_PATH)
        control_data = control_ref.get()

        if not control_data or control_data.get("complete") != True:
            print("⚠️ Trigger received, but 'complete' flag not set. Aborting.")
            return False

        # Get dataset
        data_ref = db.reference(DATA_PATH)
        data = data_ref.get()

        if not data:
            print("❌ No training data found.")
            return False

        # Save locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/training/training_data_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Training data saved to: {filename}")

        # Archive in Firebase
        archive_ref = db.reference(f"{ARCHIVE_PATH}/{timestamp}")
        archive_ref.set(data)
        print("📦 Data archived in Firebase.")

        # Extract features for ML method
        # Extract features from each batch individually
        for key, record in data.items():
            msg_id = record.get("msg_id")
            if not msg_id:
                print(f"⚠️ Skipping record without msg_id: {key}")
                continue

            extract_features_from_firebase_batch({msg_id: record})


        # Clean up original data and control flag
        data_ref.delete()
        control_ref.set({"complete": False})
        print("🧹 Training data and control flag cleared.\n")

        return True

    except Exception as e:
        print(f"❌ Exception during processing: {e}")
        return False
