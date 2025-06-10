# utils/auth.py

from supabase import Client
from typing import Tuple

def is_admin_user(user_id: int, supabase: Client) -> Tuple[bool, str]:
    """
    Check if the user is an active admin.

    Parameters:
        user_id (int): ID of the user to check.
        supabase (Client): Supabase client instance.

    Returns:
        (bool, str): Tuple where bool indicates admin status, and str is a message.
    """
    try:
        user_data = (
            supabase.table("users")
            .select("is_admin, is_active")
            .eq("id", user_id)
            .execute()
        )

        if not user_data.data:
            return False, "User not found"

        user = user_data.data[0]
        if not user.get("is_active"):
            return False, "User is inactive"
        if not user.get("is_admin"):
            return False, "User is not admin"

        return True, "User is admin"

    except Exception as e:
        return False, f"Error checking admin status: {str(e)}"
