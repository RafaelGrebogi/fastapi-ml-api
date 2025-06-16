from fastapi import FastAPI, Query
from fastapi import Request
from contextlib import asynccontextmanager
from firebase_handler import process_training_data, process_testing_data, process_production_data,  update_server_ip
from datetime import date

from fastapi.middleware.cors import CORSMiddleware
import asyncpg
from dotenv import load_dotenv
import os

from supabase import create_client, Client

load_dotenv()  # Load environment variables from .env

# app = FastAPI()



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


# # Connect to PostgreSQL each time (could be pooled later)
# async def get_connection():
#     return await asyncpg.connect(
#         host=DB_HOST,
#         port=DB_PORT,
#         user=DB_USER,
#         password=DB_PASSWORD,
#         database=DB_NAME
#     )

# @app.get("/get-user-status")
# async def get_user_status(
#     username: str = Query(...),
#     device_id: str = Query(...)
# ):
#     conn = await get_connection()
#     try:
#         result = await conn.fetchrow("""
#             SELECT 
#                 users.id,
#                 EXISTS (
#                     SELECT 1
#                     FROM service
#                     JOIN device ON service.device_id = device.id
#                     WHERE service.users_id = users.id
#                     AND device.serial_number = $2
#                     AND CURRENT_DATE BETWEEN service.start_date AND service.end_date
#                 ) AS has_active_service
#             FROM users
#             WHERE username = $1;
#         """, username, device_id)

#         if result:
#             return {
#                 "user_id": str(result["id"]),
#                 "has_active_service": result["has_active_service"]
#             }
#         else:
#             return {"error": "User not found"}
#     finally:
#         await conn.close()


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/get-user-status")
async def get_user_status(username: str = Query(...), device_id: str = Query(...)):
    try:
        # 1. Get user
        user_data = supabase.table("users").select("id, is_active").eq("username", username).execute()
        if not user_data.data or not user_data.data[0]["is_active"]:
            return {"error": "User not found"}

        user_id = user_data.data[0]["id"]
        today = str(date.today())  # e.g., '2025-05-30'

        # 2. Check service matching user AND active dates
        service_data = (
            supabase.table("service")
            .select("id, device_id")
            .eq("users_id", user_id)
            .lte("start_date", today)
            .gte("end_date", today)
            .execute()
        )

        # 3. Match device
        active = False
        if service_data.data:
            device_data = supabase.table("device").select("id").eq("serial_number", device_id).execute()
            if device_data.data:
                device_ids = [d["id"] for d in device_data.data]
                service_device_ids = [s["device_id"] for s in service_data.data]
                active = any(d in service_device_ids for d in device_ids)

        return {
            "user_id": str(user_id),
            "has_active_service": active
        }

    except Exception as e:
        return {"error": str(e)}

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

