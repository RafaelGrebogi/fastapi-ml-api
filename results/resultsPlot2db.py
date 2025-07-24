
import sys
import os
# import datetime
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from io import BytesIO

# # Add the root folder to sys.path so you can import supabase_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supabase_client import s3

# import boto3


def store_supabase_bucket(image_bytes, filename, mode, user_id):

    # ---------------------------
    # Upload to Supabase
    # ---------------------------

    # Convert to file-like object
    image_file = BytesIO(image_bytes)

    # Unique file name
    # filename = f"Training/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Define bucket and path
    bucket_name = "results-figures-bucket"  
    storage_path = f"users/{user_id}/{mode}/{filename}"  # path inside the bucket


    # Upload to S3-compatible Supabase storage
    s3.upload_fileobj(
        Fileobj=image_file,
        Bucket=bucket_name,
        Key=storage_path,
        ExtraArgs={"ContentType": "image/png", "ACL": "public-read"}
    )
    

    return True



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
    sns.kdeplot(
        x=df['accel_x_smooth'],
        y=df['accel_y_smooth'],
        fill=True,
        cmap='coolwarm',
        bw_adjust=bw_adjust,
        levels=50,
        thresh=0.01,
        ax=ax
    )
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
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

    # # Create the density plot
    # plt.figure(figsize=(8, 6))
    # sns.kdeplot(
    #     x=df['accel_x_smooth'],
    #     y=df['accel_y_smooth'],
    #     fill=True,
    #     cmap='coolwarm',
    #     bw_adjust=bw_adjust,
    #     levels=50,
    #     thresh=0.01,
    # )

    # # Add annotations
    # plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    # plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    # plt.scatter(df['accel_x_smooth'].mean(), df['accel_y_smooth'].mean(),
    #             color='black', label='Mean Position', zorder=3)

    # # Labels and title
    # plt.xlabel('accel_x (Left/Right)')
    # plt.ylabel('accel_y (Front/Back)')
    # plt.title('Smoothed Motion Density Plot')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("motion_density_plot.png")
    # print("Plot saved to file.")
    


# import json
# import sys
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Add the root folder to sys.path so you can import supabase_client
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from supabase_client import supabase, s3

# import boto3



# # ---------------------------
# # Step 1: Create dummy plot
# # ---------------------------
# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# plt.figure(figsize=(6, 4))
# plt.plot(x, y, label="sin(x)")
# plt.title("Dummy Plot")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)

# # Save as JPEG
# filename = "dummy_plot.jpg"
# plt.savefig(filename, format="jpg")
# plt.close()

# # ---------------------------
# # Step 2: Upload to Supabase
# # ---------------------------

# # Define bucket and path
# bucket_name = "results-figures-bucket"  # e.g., 'plots'
# storage_path = f"Training/{filename}"  # path inside the bucket


# # Upload file
# s3.upload_file(filename, bucket_name, storage_path)

# print("✅ Upload complete")



