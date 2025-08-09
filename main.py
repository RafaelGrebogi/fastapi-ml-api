from fastapi import FastAPI, Query
from fastapi import Request
from contextlib import asynccontextmanager
from firebase_handler import process_training_data, process_testing_data, process_production_data,  update_server_ip
from datetime import date

from fastapi.middleware.cors import CORSMiddleware
import asyncpg
from dotenv import load_dotenv
import os

from supabase_client import supabase
from utils.auth import is_admin_user
from utils.service_utils import check_service_is_active, get_device_details
from context_vars import current_user_id, current_service_id, current_DeviceSerial, current_DeviceId, current_SessionToken
from endpoints.gps_route import router as gps_router



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

app.include_router(gps_router)

@app.get("/")
def home():
    return {"message": "FastAPI ready to process data."}




@app.get("/get-user-status")
async def get_user_status(username: str = Query(...), device_id: str = Query(...)):
    try:
        # 1. Get user
        user_data = supabase.table("users").select("id, is_active").eq("username", username).execute()
        if not user_data.data or not user_data.data[0]["is_active"]:
            return {"error": "User not found"}

        user_id = user_data.data[0]["id"]
        today = str(date.today())  # e.g., '2025-06-04'




        # 2. Get all services linked to user with JOIN to company
        service_data = (
            supabase.table("service")
            .select("id, device_id, start_date, end_date, task_id, ml_method_id, company_id, company(name)")
            .eq("users_id", user_id)
            .execute()
        )

        # 3. Check for active service linked to the specified device
        active = False
        matched_service_id = None
        if service_data.data:
            device_data = supabase.table("device").select("id").eq("serial_number", device_id).execute()
            if device_data.data:
                device_ids = [d["id"] for d in device_data.data]
                for s in service_data.data:
                    if s["device_id"] in device_ids and s["start_date"] <= today <= s["end_date"]:
                        active = True
                        matched_service_id = s["id"]
                        break

        return {
            "user_id": str(user_id),
            "has_active_service": active,
            "selected_service_id": matched_service_id,
            "services": service_data.data  # now includes company.name
        }

    except Exception as e:
        return {"error": str(e)}


# -------------------------------------
# -------------------------------------

@app.get("/get-service-status")
async def get_service_status(service_id: int = Query(...)):
    try:
        today = str(date.today())

        # 1. Get the service record and join with company and user
        service_query = (
            supabase.table("service")
            .select("id, start_date, end_date, users_id, company(name)")
            .eq("id", service_id)
            .execute()
        )

        if not service_query.data:
            return {"error": "Service not found"}

        service = service_query.data[0]

        # 2. Get user_id and check if the user is active
        user_id = service["users_id"]
        user_query = (
            supabase.table("users")
            .select("id, is_active")
            .eq("id", user_id)
            .execute()
        )

        if not user_query.data or not user_query.data[0]["is_active"]:
            return {"error": "User inactive or not found"}

        # 3. Check if the service is currently active
        active = service["start_date"] <= today <= service["end_date"]

        return {
            "user_id": str(user_id),
            "service_id": str(service_id),
            "has_active_service": active,
            "company": {
                "name": service["company"]["name"]
            }
        }

    except Exception as e:
        return {"error": str(e)}


# -------------------------------------
# -------------------------------------


@app.post("/trigger-training")
async def trigger_training(request: Request):
    data = await request.json()
    device_id = data.get("device_id") # THIS IS DEVICE SERIAL NUMBER | device_id TO BE RENAMED TO device_serial
    user_id = data.get("user_id")
    service_id = data.get("service_id")
    session_token = data.get("session_token")

    if not device_id:
        return {"error": "Missing device_serial in request"}
    if not user_id:
        return {"error": "Missing user_id in request"}
    if not service_id:
        return {"error": "Missing service_id in request"}
    if not session_token:
        return {"error": "Missing session_token in request"}

    # Get admin user
    is_admin, message = is_admin_user(user_id, supabase)
    if not is_admin:
        return {"error": message}
    
    # Get service status
    is_active, message = check_service_is_active(service_id, supabase)
    if not is_active:
        return {"error": message}
    
    # Get device details
    is_correct, device_details = get_device_details(device_serial=device_id, supabase=supabase)
    if not is_correct:
        return {"error": device_details}

    # Store in ContextVars
    current_user_id.set(user_id)
    current_service_id.set(service_id)
    current_DeviceSerial.set(device_id)
    current_DeviceId.set(device_details["id"])
    current_SessionToken.set(session_token)

    success = await process_training_data(device_id=device_id)
    if success:
        return {"status": "success", "message": "Training data processed successfully."}
    else:
        return {"status": "fail", "message": "Failed to process training data."}


# -------------------------------------
# -------------------------------------


@app.post("/trigger-testing")
async def trigger_testing(request: Request):
    body = await request.json()
    
    device_id = body.get("device_id") # THIS IS DEVICE SERIAL NUMBER | device_id TO BE RENAMED TO device_serial
    user_id = body.get("user_id")
    service_id = body.get("service_id")

    # Get device details
    is_correct, device_details = get_device_details(device_serial=device_id, supabase=supabase)
    if not is_correct:
        return {"error": device_details}

    if not device_id:
        return {"error": "Missing device serial number in request"}
    if not user_id:
        return {"error": "Missing user_id in request"}
    if not service_id:
        return {"error": "Missing service_id in request"}
    
    # Store in ContextVars
    current_user_id.set(user_id)
    current_service_id.set(service_id)
    current_DeviceSerial.set(device_id)
    current_DeviceId.set(device_details["id"])
    
    result = process_testing_data(device_id)
    return result

# -------------------------------------
# -------------------------------------


@app.post("/trigger-production")
async def trigger_production(request: Request):
    body = await request.json()
    
    device_id = body.get("device_id")
    user_id = body.get("user_id")
    service_id = body.get("service_id")

    # Get device details
    is_correct, device_details = get_device_details(device_serial=device_id, supabase=supabase)
    if not is_correct:
        return {"error": device_details}
    

    if not device_id:
        return {"error": "Missing device_id in request"}
    if not user_id:
        return {"error": "Missing user_id in request"}
    if not service_id:
        return {"error": "Missing service_id in request"}
    
    # Store in ContextVars
    current_user_id.set(user_id)
    current_service_id.set(service_id)
    current_DeviceSerial.set(device_id)
    current_DeviceId.set(device_details["id"])
    
    result = process_production_data(device_id)
    return result

