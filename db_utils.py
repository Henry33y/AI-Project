import pandas as pd
import time
import os
from supabase import create_client, Client, ClientOptions
import streamlit as st
import httpx

# Initialize Supabase client with custom timeout
def init_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in environment variables or secrets.")
    
    # Increase timeout to 60s (default is often lower)
    options = ClientOptions(
        postgrest_client_timeout=60,
        storage_client_timeout=60,
    )
    return create_client(url, key, options=options)

def fetch_student_data() -> pd.DataFrame:
    """
    Fetches all student records from the 'students' table in Supabase.
    Returns a pandas DataFrame capable of being processed by model_utils.
    Includes retry logic for transient network errors.
    """
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            supabase = init_supabase()
            
            # Try to sign in as admin if credentials are provided
            admin_email = os.environ.get("ADMIN_EMAIL") or st.secrets.get("ADMIN_EMAIL")
            admin_password = os.environ.get("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD")
            
            if admin_email and admin_password:
                try:
                    supabase.auth.sign_in_with_password({
                        "email": admin_email,
                        "password": admin_password
                    })
                except Exception as e:
                    # Only print warning on first failure to avoid log spam
                    if attempt == 0:
                        print(f"Warning: Admin login failed: {e}. Proceeding with anon key.")

            # Fetch data
            response = supabase.table("students").select("*").execute()
            data = response.data
            
            if not data:
                return pd.DataFrame()
                
            return pd.DataFrame(data)

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Connection failed ({e}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise e
    
    return pd.DataFrame()
