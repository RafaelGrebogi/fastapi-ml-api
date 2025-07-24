# supabase_client.py
import os
from supabase import create_client, Client

import boto3

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SUPABASE_DB_BUCKET_ACCESS_KEY_ID = os.getenv("SUPABASE_DB_BUCKET_ACCESS_KEY_ID")
SUPABASE_DB_BUCKET_SECRET_ACCESS_KEY = os.getenv("SUPABASE_DB_BUCKET_SECRET_ACCESS_KEY")
SUPABASE_DB_BUCKET_END_POINT = os.getenv("SUPABASE_DB_BUCKET_END_POINT")
SUPABASE_DB_BUCKET_REGION = os.getenv("SUPABASE_DB_BUCKET_REGION")

s3 = boto3.client(
    "s3",
    endpoint_url=SUPABASE_DB_BUCKET_END_POINT,
    aws_access_key_id=SUPABASE_DB_BUCKET_ACCESS_KEY_ID,
    aws_secret_access_key=SUPABASE_DB_BUCKET_SECRET_ACCESS_KEY,
    region_name=SUPABASE_DB_BUCKET_REGION,
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
