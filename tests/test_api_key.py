from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("BIRDEYE_API_KEY")

if api_key:
    print("API key loaded successfully")
else:
    print("API key not found")