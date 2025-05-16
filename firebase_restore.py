import firebase_admin
from firebase_admin import credentials, db
import os
import json
from datetime import datetime
import socket

# Firebase setup
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://esp32-datalogger-c9c32-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

# Paths
TRAINING_CONTROL_PATH = "/ESP32_Develop/ControlFlag/"
TRAINING_DATA_PATH = "/ESP32_Develop/TrainingDataset/"
TRAINING_ARCHIVE_PATH = "/ESP32_Develop/TrainingArchive/"

FIREBASE_TESTING_PATH = "/ESP32_Develop/TestingDataset/"


PRODUCTION_CONTROL_PATH = "/ESP32_Production/ControlFlag/"
PRODUCTION_DATA_PATH = "/ESP32_Production/Data/"
PRODUCTION_ARCHIVE_PATH = "/ESP32_Production/Archive/"

def list_archives(archive_base_path: str):
    """
    List all available archives under a given base path.

    Parameters:
        archive_base_path (str): The root path of the archives.

    Returns:
        list: A list of archive paths.
    """
    try:
        base_ref = db.reference(archive_base_path)
        archives = base_ref.get()

        if not archives:
            print(f"⚠️ No archives found under path: {archive_base_path}")
            return []

        archive_list = []
        for device_id, timestamps in archives.items():
            for timestamp in timestamps:
                archive_path = f"{archive_base_path}{device_id}/{timestamp}"
                archive_list.append(archive_path)

        print(f"✅ Found {len(archive_list)} archives.")
        return archive_list

    except Exception as e:
        print(f"❌ Error listing archives: {e}")
        return []


def unarchive_data_from_firebase(archive_path: str, restore_path: str, del_Archive: bool = False) -> bool:
    """
    Unarchive data from Firebase and restore it to another Firebase path.

    Parameters:
        archive_path (str): The Firebase path where the data is archived.
        restore_path (str): The Firebase path where the data should be restored.
        del_Archive (bool): Whether to delete the archive after restoring.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Access the archived data from Firebase
        archive_ref = db.reference(archive_path)
        archived_data = archive_ref.get()

        if not archived_data:
            print(f"❌ No data found at archive path: {archive_path}")
            return False

        # Set the data to the restore path in Firebase
        restore_ref = db.reference(restore_path)
        restore_ref.set(archived_data)
        print(f"✅ Data restored from '{archive_path}' to '{restore_path}' in Firebase.")

        # Delete the archive if the flag is set to True
        if del_Archive:
            archive_ref.delete()
            print(f"🗑️ Archive deleted from Firebase: {archive_path}")
        else:
            print(f"📂 Archive kept in Firebase: {archive_path}")

        return True

    except Exception as e:
        print(f"❌ Error during unarchiving: {e}")
        return False


def main():
    # List available archives
    archive_list = list_archives(PRODUCTION_ARCHIVE_PATH)

    if not archive_list:
        print("❌ No archives available to restore.")
        return

    # Select the latest archive (sorted by timestamp)
    latest_archive = sorted(archive_list)[-1]
    print(f"ℹ️ Latest archive found: {latest_archive}")

    # User choice: whether to delete the archive after restoring
    del_Archive = input("Delete archive after restoring? (yes/no): ").strip().lower() == "yes"

    # Attempt to restore the latest archive
    if unarchive_data_from_firebase(latest_archive, PRODUCTION_DATA_PATH, del_Archive):
        print("✅ Data successfully restored.")
    else:
        print("❌ Failed to restore data.")



if __name__ == "__main__":
    main()
