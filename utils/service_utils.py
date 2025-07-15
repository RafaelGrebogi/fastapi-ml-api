# utils/service_details.py

from supabase import Client
from typing import Tuple, Union, Dict
import os
import json
from context_vars import current_user_id, current_service_id, current_DeviceId





def get_device_details(device_serial: str, supabase: Client) -> Tuple[bool, Union[dict, str]]:
    try:
        response = (
            supabase.table("device")
            .select("id, serial_number")
            .eq("serial_number", device_serial)
            .execute()
        )

        if not response.data:
            return False, "Device not found"

        device = response.data[0]

        
        return True, device

    except Exception as e:
        return False, f"Error fetching device details: {str(e)}"






def get_service_details(service_id: int, supabase: Client) -> Tuple[bool, Union[dict, str]]:
    """
    Retrieve full service configuration including task and ml_method.

    Parameters:
        service_id (int): ID of the service to load.
        supabase (Client): Supabase client instance.

    Returns:
        Tuple[bool, dict or str]: (Success, Data or Error Message)
    """
    try:
        response = (
            supabase.table("service")
            .select("id, device_id, users_id, start_date, end_date, task(*), ml_method(*)")
            .eq("id", service_id)
            .execute()
        )

        if not response.data:
            return False, "Service not found"

        service = response.data[0]

        # Optional: validate date range, task or ml_method existence
        if not service.get("ml_method") or not service.get("task"):
            return False, "Service config incomplete"

        return True, service

    except Exception as e:
        return False, f"Error fetching service details: {str(e)}"
    


def check_service_is_active(service_id: int, supabase: Client) -> Tuple[bool, str]:
    """
    Check if the service exists and is active (based on start_date and end_date).

    Parameters:
        service_id (int): ID of the service to verify.
        supabase (Client): Supabase client instance.

    Returns:
        Tuple[bool, str]: (True, message) if active; (False, error message) otherwise.
    """
    try:
        # Query the service table
        response = (
            supabase.table("service")
            .select("id, start_date, end_date")
            .eq("id", service_id)
            .execute()
        )

        if not response.data:
            return False, "Service not found"

        service = response.data[0]

        from datetime import date

        today = date.today()
        start_date = date.fromisoformat(service["start_date"])
        end_date = date.fromisoformat(service["end_date"])

        if start_date <= today <= end_date:
            return True, "Service is active"
        else:
            return False, f"Service is inactive (valid from {start_date} to {end_date})"

    except Exception as e:
        return False, f"Error checking service status: {str(e)}"





def upload_result_to_db(json_data: Dict, supabase: Client, isDev: bool, mode: int) -> Dict:
    try:
        user_id = current_user_id.get()
        service_id = current_service_id.get()
        DeviceId = current_DeviceId.get()
        response = supabase.table("results").insert({
            "user_id": user_id,
            "service_id": service_id,
            "is_dev": isDev,
            "device_id": DeviceId,
            "mode_id": mode,
            "result_json": json_data  
        }).execute()

        if response.data:
            return {"success": True, "message": "Result uploaded successfully", "data": response.data}
        else:
            return {"success": False, "message": "Insert failed", "error": response.data}
    except Exception as e:
        return {"success": False, "message": "Exception occurred", "error": str(e)}