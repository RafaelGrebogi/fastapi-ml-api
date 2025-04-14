# Re-import necessary packages after code state reset
import os
import json
import pandas as pd
from pathlib import Path

# Ensure the working environment is correct and re-import custom module
# Copy the file into a usable directory for testing
source_path = "/mnt/c/Users/rafag/ml-api/data/esp32-rtdb-export.json"
test_file_path = "/mnt/c/Users/rafag/ml-api/data/esp32_rtdb_filtered.json"

# Load and filter the input file
with open(source_path, "r") as f:
    full_data = json.load(f)

# Navigate into the nested 'TrainingDataset' branch
training_data = full_data.get("ESP32_Develop", {}).get("TrainingDataset", {})

# Filter for messages with exactly 10 samples
filtered_data = {
    msg_id: msg
    for msg_id, msg in training_data.items()
    if len(msg.get("samples", [])) == 10
}

# Save filtered test data for reference
with open(test_file_path, "w") as f:
    json.dump(filtered_data, f, indent=2)

# Confirm how many valid messages we found
{
    "filtered_message_count": len(filtered_data),
    "saved_test_path": test_file_path
}
