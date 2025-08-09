
import sys
import os
# import datetime
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from io import BytesIO
from supabase import Client
from typing import Dict


# # Add the root folder to sys.path so you can import supabase_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supabase_client import s3, supabase
from context_vars import current_user_id, current_service_id, current_result_id, current_SessionToken

# import boto3


# --------------------------------
# --------------------------------

def plot_motion_density(data, window_size=5, bw_adjust=0.5):
    """
    Plots a density map of smoothed accel_x (left/right) vs accel_y (front/back) data.

    Parameters:
        data (list of dicts): Raw sensor data with 'accel_x' and 'accel_y' keys.
        window_size (int): Window size for moving average smoothing.
        bw_adjust (float): Bandwidth adjustment for the KDE smoothing.

            Returns:
        bytes: PNG image data ready for uploading.
    """

    full_sample_list = []

    for msg_id, content in data.items():
        samples = content.get("samples", [])
        if not samples:
            print(f"⚠️ No samples found in message: {msg_id}")
            continue
        full_sample_list.extend(samples)

    if len(full_sample_list) < window_size:
        print(f"⚠️ Not enough combined samples: {len(full_sample_list)}")
        return False, None

    df = pd.DataFrame(full_sample_list)

    # Ensure required keys are present
    if 'accel_x' not in df or 'accel_y' not in df:
        raise KeyError("Data must contain 'accel_x' and 'accel_y'.")

    # Apply moving average smoothing
    df['accel_x_smooth'] = uniform_filter1d(df['accel_x'], size=window_size)
    df['accel_y_smooth'] = uniform_filter1d(df['accel_y'], size=window_size)

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot KDE and store the object to attach colorbar
    kde = sns.kdeplot(
        x=df['accel_x_smooth'],
        y=df['accel_y_smooth'],
        fill=True,
        cmap='coolwarm',
        bw_adjust=bw_adjust,
        levels=50,
        thresh=0.01,
        ax=ax
    )

    # Add colorbar for density scale
    cbar = plt.colorbar(kde.collections[0], ax=ax, label="Relative Density")

    # Create the figure and plot
    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.kdeplot(
    #     x=df['accel_x_smooth'],
    #     y=df['accel_y_smooth'],
    #     fill=True,
    #     cmap='coolwarm',
    #     bw_adjust=bw_adjust,
    #     levels=50,
    #     thresh=0.01,
    #     ax=ax
    # )


    # ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    # ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    # Highlighted origin axis
    ax.axhline(0, color='black', linestyle='-', linewidth=2, zorder=2)
    ax.axvline(0, color='black', linestyle='-', linewidth=2, zorder=2)
    
    ax.scatter(df['accel_x_smooth'].mean(), df['accel_y_smooth'].mean(),
               color='black', label='Mean Position', zorder=3)
    ax.set_xlabel('accel_x (Left/Right)')
    ax.set_ylabel('accel_y (Front/Back)')
    ax.set_title('Smoothed Motion Density Plot')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Save to memory buffer
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    plt.close(fig)
    buffer.seek(0)
    
    return True, buffer.getvalue()




# --------------------------------
# --------------------------------

def store_supabase_bucket(image_bytes, filename, mode, figure_type):

    # ---------------------------
    # Upload to Supabase
    # ---------------------------

    # Convert to file-like object
    image_file = BytesIO(image_bytes)

    user_id = current_user_id.get()
    service_id = current_service_id.get()


    # Define bucket and path
    bucket_name = "results-figures-bucket"  
    project_ref = "jptmqikyuuxavclmlrhw" # https://jptmqikyuuxavclmlrhw.storage.supabase.co/storage/v1/s3
    storage_path = f"users/{user_id}/{mode}/Service{service_id}/{filename}"  # path inside the bucket


    # Upload to S3-compatible Supabase storage
    s3.upload_fileobj(
        Fileobj=image_file,
        Bucket=bucket_name,
        Key=storage_path,
        ExtraArgs={"ContentType": "image/png", "ACL": "public-read"}
    )
    
    response = upload_figure_url_to_db(
        bucket=bucket_name,
        path=storage_path,
        project_ref=project_ref,
        supabase=supabase,
        figure_type=figure_type
    )

    if response["success"]:
        print("✅ Upload successful!")
        print("Figure URL:", response["url"])
    else:
        print("❌ Upload failed:", response["message"])
        return False

    return True





# --------------------------------
# --------------------------------

def upload_public_url(bucket: str, path: str, project_ref: str) -> str:
    url_str = f"https://{project_ref}.supabase.co/storage/v1/object/public/{bucket}/{path}"
    
    return url_str




# def upload_figure_url_to_db(bucket: str, path: str, project_ref: str, supabase: Client, figure_type: str) -> Dict:
#     try:
#         result_id = current_result_id.get()

#         if not result_id:
#             return {"success": False, "message": "No result_id found in context"}

#         # Construct the public URL
#         public_url = upload_public_url(bucket=bucket, path=path, project_ref=project_ref)
#         figure_url_json = {figure_type: public_url}

#         # Perform the update
#         response = supabase.table("results").update({
#             "figure_url": figure_url_json
#         }).eq("id", result_id).execute()

#         if response.data:
#             return {
#                 "success": True,
#                 "message": "Figure URL updated successfully",
#                 "url": figure_url_json,
#                 "data": response.data[0]
#             }
#         else:
#             return {
#                 "success": False,
#                 "message": "Update failed",
#                 "error": response.data
#             }

#     except Exception as e:
#         return {
#             "success": False,
#             "message": "Exception occurred while uploading figure URL",
#             "error": str(e)
#         }

# --------------------------------
# --------------------------------

def upload_figure_url_to_db(bucket: str, path: str, project_ref: str, supabase: Client, figure_type: str) -> Dict:
    try:
        session_token = current_SessionToken.get()
        if session_token is None:
            return {"success": False, "message": "No current session_token set."}

        # Build new public URL
        public_url = upload_public_url(bucket=bucket, path=path, project_ref=project_ref)

        # Step 1: Get existing figure_url field
        existing_response = supabase.table("results").select("figure_url").eq("session_token", session_token).execute()

        if not existing_response.data:
            return {"success": False, "message": "Result not found"}

        existing_figure_url = existing_response.data[0].get("figure_url", {})

        # Step 2: Merge with new figure type
        if not isinstance(existing_figure_url, dict):
            existing_figure_url = {}

        existing_figure_url[figure_type] = public_url

        # Step 3: Update DB with merged figure_url JSON
        update_response = supabase.table("results").update({
            "figure_url": existing_figure_url
        }).eq("session_token", session_token).execute()

        if update_response.data:
            return {
                "success": True,
                "message": "Figure URL added successfully",
                "url": public_url
            }
        else:
            return {"success": False, "message": "Update failed", "error": update_response.data}

    except Exception as e:
        return {"success": False, "message": "Exception occurred", "error": str(e)}



