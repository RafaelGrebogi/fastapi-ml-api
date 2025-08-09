from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict


import os
import sys

# # Add the root folder to sys.path so you can import supabase_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supabase_client import supabase



class GPSCoordinates(BaseModel):
    session_token: str
    gps_coordinates: Dict[str, str]  


# Define router
router = APIRouter()


@router.post("/upload-gps")
async def upload_gps_coordinates(data: GPSCoordinates):
    try:
        response = supabase.table("results").update({
            "gps_coordinates": data.gps_coordinates
        }).eq("session_token", data.session_token).execute()

        if response.data:
            return {"success": True, "message": "GPS coordinates updated", "data": response.data[0]}
        else:
            return {"success": False, "message": "Update failed", "error": response.data}

    except Exception as e:
        return {"success": False, "message": "Exception occurred", "error": str(e)}
