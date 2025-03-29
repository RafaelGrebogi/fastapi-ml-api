from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
import firebase_admin
from firebase_admin import credentials, db

# --- Firebase setup ---
cred = credentials.Certificate("firebase_key.json")  # Your service account key
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://esp32-datalogger-c9c32-default-rtdb.asia-southeast1.firebasedatabase.app/"  # Replace with your Firebase URL
})

# --- Background task: Firebase polling ---
async def poll_firebase():
    while True:
        try:
            ref = db.reference("/ESP32_Develop/TrainingDataset/")  # Replace with your Firebase path
            data = ref.get()
            if data:
                print("📥 New data from Firebase:", data)

                # Optional: clear data after processing
                # ref.delete()

        except Exception as e:
            print("⚠️ Error while polling Firebase:", e)

        await asyncio.sleep(10)  # Poll every 10 seconds

# --- FastAPI lifespan setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(poll_firebase())  # Start the background polling task
    yield
    print("🔚 FastAPI is shutting down.")

# --- FastAPI app ---
app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "FastAPI with Firebase polling is running"}
