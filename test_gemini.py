import google.generativeai as genai
import toml
import os

try:
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets.get("GEMINI_API_KEY")
    print(f"Testing API Key: {api_key[:10]}...")
    
    genai.configure(api_key=api_key)
    print("Listing models...")
    models = list(genai.list_models())
    print("Success! Found models:", [m.name for m in models[:3]])
except Exception as e:
    print("\nError verifying API key:")
    print(e)
