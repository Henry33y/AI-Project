import os
import streamlit as st
from db_utils import fetch_student_data

# Mock secrets if running locally without streamlit run
import toml
try:
    secrets = toml.load(".streamlit/secrets.toml")
    os.environ["SUPABASE_URL"] = secrets.get("SUPABASE_URL", "")
    os.environ["SUPABASE_KEY"] = secrets.get("SUPABASE_KEY", "")
    os.environ["ADMIN_EMAIL"] = secrets.get("ADMIN_EMAIL", "")
    os.environ["ADMIN_PASSWORD"] = secrets.get("ADMIN_PASSWORD", "")
    print("Loaded secrets from .streamlit/secrets.toml")
except Exception as e:
    print(f"Could not load secrets.toml: {e}")
    print("relying on environment variables...")

try:
    print("Attempting to fetch data...")
    df = fetch_student_data()
    print("Success!")
    print(f"Retrieved {len(df)} records.")
    print("Columns:", df.columns.tolist())
    print(df.head())
except Exception as e:
    print(f"Error fetching data: {e}")
