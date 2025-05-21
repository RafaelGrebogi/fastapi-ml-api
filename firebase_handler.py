import firebase_admin
from firebase_admin import credentials, db
import os
import json
from datetime import datetime
# import asyncio
import socket
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

async def process_training_data(device_id: str):
    try:
        data, data_ref = download_data_if_complete(
            firebase_data_path=DATA_PATH,
            firebase_control_path=CONTROL_PATH,
            local_dir=TRAINING_DATA_DIR,
            archive_dir=ARCHIVE_PATH,
            EXPECTED_DEVICE_ID=device_id  #  new parameter
        )

        # # Small precaution delay in case ESP32 is still writing
        # await asyncio.sleep(2)

        # control_ref = db.reference(CONTROL_PATH)
        # control_data = control_ref.get()

        # if not control_data or control_data.get("complete") != True:
        #     print("❌ Trigger received, but 'complete' flag not set. Aborting.")
        #     return False

        # # Get dataset
        # data_ref = db.reference(DATA_PATH)
        # data = data_ref.get()

        # if not data:
        #     print("❌ No training data found.")
        #     return False

        # # Save locally
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"data/training/training_data_{timestamp}.json"
        # with open(filename, "w") as f:
        #     json.dump(data, f, indent=4)
        # print(f"💾 Training data saved to: {filename}")

        # # Archive in Firebase
        # archive_ref = db.reference(f"{ARCHIVE_PATH}/{timestamp}")
        # archive_ref.set(data)
        # print("📦 Data archived in Firebase.")

        # Extract features for ML method
        # Extract features from each batch individually

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
def process_testing_data(device_id: str):
    print(" Triggered: Testing Mode")

    data, _ = download_data_if_complete(
        FIREBASE_TESTING_PATH,
        CONTROL_PATH,
        TESTING_DATA_DIR,
        ARCHIVE_PATH,
        device_id
    )

    csv_path = extract_features_from_firebase_batch(data, USE_MULTI_MESSAGE_WINDOW=True)
    result = run_ml_pipeline("testing", csv_path)
    return result

# ===========================================
# ===========================================
def process_production_data(device_id: str):
    print(" Triggered: Production Mode")

    data, _ = download_data_if_complete(
        FIREBASE_PRODUCTION_PATH,
        CONTROL_PATH,
        PRODUCTION_DATA_DIR,
        ARCHIVE_PATH,
        device_id
    )

    csv_path = extract_features_from_firebase_batch(data, USE_MULTI_MESSAGE_WINDOW=True)
    result = run_ml_pipeline("production", csv_path)
    return result

# ===========================================
# ===========================================

# def download_data_if_complete(firebase_data_path: str, firebase_control_path: str, local_dir: str, archive_dir: str) -> str:

def get_local_ip():
    """Return the fixed Windows IP address."""
    local_ip = "192.168.20.5"
    print(f"🔧 Using hard-coded Windows IP: {local_ip}")
    return local_ip

# def get_local_ip():
#     """Get the actual local IP address."""
#     try:
#         # Create a socket to an external address to determine the local IP
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         # Connect to a public DNS server (Google's)
#         s.connect(("8.8.8.8", 80))
#         local_ip = s.getsockname()[0]
#         s.close()
#         return local_ip
#     except Exception as e:
#         print(f"❌ Failed to get local IP: {e}")
#         return None

def update_server_ip():
    """Update the server IP address in Firebase."""
    local_ip = get_local_ip()
    if local_ip:
        try:
            ref = db.reference("/ServerIP")
            ref.set(local_ip)
            print(f"✅ Updated FastAPI server IP to: {local_ip}")
        except Exception as e:
            print(f"❌ Failed to update IP in Firebase: {e}")
    else:
        print("⚠️ No local IP obtained. Skipping Firebase update.")

# ===========================================
# ===========================================


def download_data_if_complete(firebase_data_path, firebase_control_path, local_dir, archive_dir, EXPECTED_DEVICE_ID)-> tuple:
    """
    Checks the 'complete' flag in Firebase, downloads the dataset if ready,
    saves it locally, archives it, and deletes it from the original path.

    """
    import time
    # from datetime import datetime
    from pathlib import Path

    

    # Delay to ensure ESP32 has finished uploading
    time.sleep(2)

    # control_ref = db.reference(firebase_control_path)
    control_ref = db.reference(f"{firebase_control_path}{EXPECTED_DEVICE_ID}")
    control_data = control_ref.get()

    if not control_data or control_data.get("complete") != True:
        print("❌ Trigger received, but 'complete' flag not set. Aborting.")
        raise RuntimeError("Control flag not set to 'complete'.")

    # Download full data
    data_ref = db.reference(firebase_data_path)
    data = data_ref.get()

    if not data:
        print("❌ No data found at Firebase path.")
        raise RuntimeError("No data found in Firebase.")

    # Split into matching and unmatched device_ids
    matching_data = {k: v for k, v in data.items() if v.get("device_id") == EXPECTED_DEVICE_ID}
    unmatched_data = {k: v for k, v in data.items() if v.get("device_id") != EXPECTED_DEVICE_ID}

    if not matching_data:
        print("⚠️ No matching device_id found. Returning data to Firebase.")
        data_ref.set(data)  # Restore original dataset
        raise RuntimeError("No data matched expected device_id. Skipping processing.")

    # Archive only matching data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    archive_ref = db.reference(f"{archive_dir}/{EXPECTED_DEVICE_ID}/{timestamp}")
    archive_ref.set(matching_data)
    print(" Matching data archived in Firebase.")


    # Save matching data locally
    filename = f"data_{timestamp}.json"
    save_path = Path(local_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(matching_data, f, indent=2)

    print(f" Matching Firebase data saved to: {save_path}")


    # Replace original path with unmatched entries (or clear it)
    if unmatched_data:
        data_ref.set(unmatched_data)

        print(" Unmatched data restored to Firebase.")
    else:
        data_ref.delete()
        print(" All data processed and deleted from Firebase.")

    return matching_data, data_ref

