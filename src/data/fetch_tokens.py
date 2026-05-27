"""
fetch_tokens.py

Purpose:
    Fetch a small batch of recently listed Solana tokens from Birdeye
    and save the raw token discovery data locally.

Important idea:
    This file should NOT train a model.
    This file should NOT build final features.
    This file only collects raw token/token-snapshot data.

Expected output:
    data/raw/tokens_discovered.csv
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------

# This loads variables from your .env file.
# Example .env:
# BIRDEYE_API_KEY=your_actual_api_key_here
load_dotenv()

API_KEY = os.getenv("BIRDEYE_API_KEY")

if API_KEY is None:
    raise ValueError(
        "BIRDEYE_API_KEY was not found. "
        "Make sure it is saved in your .env file."
    )


# ---------------------------------------------------------
# 2. Set up the Birdeye request
# ---------------------------------------------------------

# Birdeye Token List V3 endpoint.
# This returns a list of tokens on the selected chain.
URL = "https://public-api.birdeye.so/defi/v3/token/list"

headers = {
    "accept": "application/json",
    "X-API-KEY": API_KEY,
    "x-chain": "solana",
}

# These parameters are intentionally simple for your first dataset.
# We sort by recent_listing_time so we are looking at newer tokens.
# limit=100 means one request can return up to 100 tokens.
params = {
    "sort_by": "recent_listing_time",
    "sort_type": "desc",
    "limit": 100,
    "offset": 0,

    # Basic filters to avoid completely dead/empty tokens.
    # You can loosen or tighten these later.
    "min_liquidity": 1000,
    "min_volume_24h_usd": 100,
    "min_trade_24h_count": 5,
}


# ---------------------------------------------------------
# 3. Make the API request
# ---------------------------------------------------------

response = requests.get(URL, headers=headers, params=params, timeout=30)

print(f"Status code: {response.status_code}")

# If the request fails, this will stop the script and show the error.
response.raise_for_status()

data = response.json()


# ---------------------------------------------------------
# 4. Extract the token list from the response
# ---------------------------------------------------------

# Birdeye responses usually put the useful data inside the "data" field.
raw_data = data.get("data")

if raw_data is None:
    raise ValueError("No 'data' field found in the Birdeye response.")

# Depending on the exact response shape, the token list may be directly
# inside data or inside a nested field like data["items"].
if isinstance(raw_data, list):
    tokens = raw_data
elif isinstance(raw_data, dict):
    tokens = raw_data.get("items", [])
else:
    raise ValueError("Unexpected response format from Birdeye.")

if len(tokens) == 0:
    print("No tokens returned. You may need to loosen your filters.")


# ---------------------------------------------------------
# 5. Convert to a DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(tokens)

# Add your own collection timestamp.
# This is important because it tells you when YOU discovered/observed the token.
collected_at = datetime.now(timezone.utc).isoformat()
df["collected_at_utc"] = collected_at


# ---------------------------------------------------------
# 6. Save raw data outside src/
# ---------------------------------------------------------

# This assumes your script is located at:
# src/data/fetch_tokens.py
#
# parents[0] = src/data
# parents[1] = src
# parents[2] = project root
project_root = Path(__file__).resolve().parents[2]

raw_data_dir = project_root / "data" / "raw"
raw_data_dir.mkdir(parents=True, exist_ok=True)

output_path = raw_data_dir / "tokens_discovered.csv"

df.to_csv(output_path, index=False)

print(f"Saved {len(df)} tokens to: {output_path}")


# ---------------------------------------------------------
# 7. Quick preview
# ---------------------------------------------------------

# This lets you quickly see what columns Birdeye gave you.
print("\nColumns returned:")
print(df.columns.tolist())

print("\nFirst few rows:")
print(df.head())