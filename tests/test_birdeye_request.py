import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("BIRDEYE_API_KEY")

if not api_key:
    raise ValueError("BIRDEYE_API_KEY not found in .env")

url = "https://public-api.birdeye.so/defi/price"

headers = {
    "accept": "application/json",
    "X-API-KEY": api_key,
    "x-chain": "solana",
}

params = {
    # Wrapped SOL token address
    "address": "So11111111111111111111111111111111111111112"
}

response = requests.get(url, headers=headers, params=params)

print("Status code:", response.status_code)
print(response.json())