from fastapi import FastAPI, Query
from fastapi import Request
from contextlib import asynccontextmanager
from firebase_handler import process_training_data, process_testing_data, process_production_data,  update_server_ip

from fastapi.middleware.cors import CORSMiddleware
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

app = FastAPI()



# -------------------------------------
# -------------------------------------
# Read env vars
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_PORT = os.getenv("SUPABASE_DB_PORT", 6543)
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")




@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🌐 FastAPI starting up...")  # Confirmation message
    update_server_ip()  # Call the IP update function during startup
    print("✅ IP updated successfully.")  # Confirmation message
    yield
    print("🔚 FastAPI is shutting down.")

app = FastAPI(lifespan=lifespan)



@app.get("/")
def home():
    return {"message": "FastAPI ready to process data."}


# Connect to PostgreSQL each time (could be pooled later)
async def get_connection():
    return await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

@app.get("/get-user-id")
async def get_user_id(username: str = Query(...)):
    conn = await get_connection()
    try:
        user = await conn.fetchrow('SELECT id FROM users WHERE "Name" = $1', username)
        if user:
            return {"user_id": str(user["id"])}
        else:
            return {"error": "User not found"}
    finally:
        await conn.close()


# -------------------------------------
# -------------------------------------





@app.post("/trigger-training")
async def trigger_training(request: Request):
    data = await request.json()
    device_id = data.get("device_id")

    if not device_id:
        return {"error": "Missing device_id in request"}

    success = await process_training_data(device_id=device_id)
    if success:
        return {"status": "success", "message": "Training data processed successfully."}
    else:
        return {"status": "fail", "message": "Failed to process training data."}


@app.post("/trigger-testing")
async def trigger_testing(request: Request):
    body = await request.json()
    device_id = body.get("device_id")
    if not device_id:
        return {"error": "Missing device_id in request"}
    
    result = process_testing_data(device_id)
    return result

@app.post("/trigger-production")
async def trigger_production(request: Request):
    body = await request.json()
    device_id = body.get("device_id")
    if not device_id:
        return {"error": "Missing device_id in request"}
    
    result = process_production_data(device_id)
    return result

