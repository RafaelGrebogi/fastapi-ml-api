from fastapi import FastAPI
from firebase_handler import process_training_data

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI ready to process training data."}

@app.post("/trigger-training")
async def trigger_training():
    success = await process_training_data()
    if success:
        return {"status": "success", "message": "Training data processed successfully."}
    else:
        return {"status": "fail", "message": "Failed to process training data."}



#----------------------------------
# from fastapi import FastAPI
# from contextlib import asynccontextmanager
# import asyncio
# import firebase_admin
# from firebase_admin import credentials, db
# import os
# import json
# from datetime import datetime

# # --- Firebase setup ---
# cred = credentials.Certificate("firebase_key.json")  # Your service account key
# firebase_admin.initialize_app(cred, {
#     "databaseURL": "https://esp32-datalogger-c9c32-default-rtdb.asia-southeast1.firebasedatabase.app/"  # Replace with your Firebase URL
# })

# # --- Background task: Firebase polling ---
# async def poll_firebase():
#     os.makedirs("data/debug", exist_ok=True)  # Ensure save folder exists

#     while True:
#         try:
#             ref = db.reference("/ESP32_Develop/TrainingDataset/")  # Replace with your Firebase path
#             data = ref.get()
#             if data:
#                 # Create a timestamped filename
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"data/debug/firebase_data_{timestamp}.json"

#                 # Save the data to a new file
#                 with open(filename, "w") as f:
#                     json.dump(data, f, indent=4)

#                 print(f"💾 Data saved to {filename}")
#                 # print("📥 New data from Firebase:", data)

#                 # Optional: clear data after processing
#                 # ref.delete()

#         except Exception as e:
#             print("⚠️ Error while polling Firebase:", e)

#         await asyncio.sleep(10)  # Poll every 10 seconds

# # --- FastAPI lifespan setup ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     asyncio.create_task(poll_firebase())  # Start the background polling task
#     yield
#     print("🔚 FastAPI is shutting down.")

# # --- FastAPI app ---
# app = FastAPI(lifespan=lifespan)

# @app.get("/")
# def home():
#     return {"message": "FastAPI with Firebase polling is running"}
