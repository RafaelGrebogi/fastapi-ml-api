import firebase_admin
from firebase_admin import credentials, db
import os

# Firebase setup
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://esp32-datalogger-c9c32-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

# Paths to clean
PATHS_TO_DELETE = [
    "/ESP32_Develop/TrainingDataset/",
    "/ESP32_Develop/TrainingArchive/",
    "/ESP32_Develop/TestingDataset/",
    "/ESP32_Develop/ControlFlag/",
    "/ServerIP/"
    # Add more paths here if needed
]

def delete_firebase_path(path: str):
    """
    Delete all data under the given Firebase path.

    Parameters:
        path (str): The Firebase path to delete.
    """
    try:
        ref = db.reference(path)
        data = ref.get()

        if data is None:
            print(f"ℹ Path '{path}' is already empty.")
            return

        ref.delete()
        print(f" Successfully deleted all data at: {path}")

    except Exception as e:
        print(f"❌ Error deleting path '{path}': {e}")

def main():
    print("⚠️ WARNING: This script will permanently delete data from the specified Firebase paths.")
    confirm = input("Are you sure you want to continue? Type 'yes' to confirm: ").strip().lower()

    if confirm != 'yes':
        print("❌ Deletion cancelled.")
        return

    for path in PATHS_TO_DELETE:
        delete_firebase_path(path)

    print("✅ Cleanup completed.")

if __name__ == "__main__":
    main()
