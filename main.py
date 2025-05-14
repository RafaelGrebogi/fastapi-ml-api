from fastapi import FastAPI
from fastapi import Request
from contextlib import asynccontextmanager
from firebase_handler import process_training_data, process_testing_data, process_production_data,  update_server_ip

app = FastAPI()


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

