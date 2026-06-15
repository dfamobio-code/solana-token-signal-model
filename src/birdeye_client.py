"""
birdeye_client.py

Shared Birdeye API helper functions.

This file centralizes:
    - API key loading
    - request headers
    - global request pacing
    - retry logic
    - common Birdeye endpoint helpers
"""

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

# Load .env from the project root.
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------
# API setup
# ---------------------------------------------------------------------

BIRDEYE_BASE_URL = "https://public-api.birdeye.so"

BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")

if not BIRDEYE_API_KEY:
    raise ValueError(
        "BIRDEYE_API_KEY was not found. "
        "Make sure .env exists in the project root."
    )


# Wrapped SOL mint address.
WRAPPED_SOL_ADDRESS = "So11111111111111111111111111111111111111112"


# ---------------------------------------------------------------------
# Global request pacing
# ---------------------------------------------------------------------

# This controls how much time must pass between ANY two Birdeye requests.
# It is intentionally global so token_overview, price, and holder requests
# are all spaced out from one another.
REQUEST_INTERVAL_SECONDS = 0.0

# Last request timestamp.
LAST_REQUEST_TIME = 0.0


def set_request_interval(seconds: float) -> None:
    """
    Set the minimum delay between Birdeye API requests.

    Example:
        set_request_interval(3)

    means every Birdeye request will wait until at least 3 seconds have
    passed since the previous Birdeye request.
    """

    global REQUEST_INTERVAL_SECONDS

    REQUEST_INTERVAL_SECONDS = max(0.0, float(seconds))


def wait_for_request_slot() -> None:
    """
    Wait until it is safe to make the next Birdeye request.

    This prevents back-to-back endpoint calls from happening too quickly.
    """

    global LAST_REQUEST_TIME

    if REQUEST_INTERVAL_SECONDS <= 0:
        return

    now = time.monotonic()
    elapsed = now - LAST_REQUEST_TIME
    remaining = REQUEST_INTERVAL_SECONDS - elapsed

    if remaining > 0:
        print(f"Waiting {remaining:.2f} seconds before next Birdeye request...")
        time.sleep(remaining)


def mark_request_finished() -> None:
    """
    Record the time when a Birdeye request finished.
    """

    global LAST_REQUEST_TIME

    LAST_REQUEST_TIME = time.monotonic()


# ---------------------------------------------------------------------
# Request helper
# ---------------------------------------------------------------------

def birdeye_get(
    endpoint: str,
    params: dict | None = None,
    max_retries: int = 6,
    retry_waits: list[int] | None = None,
) -> dict:
    """
    Send a GET request to Birdeye with pacing and retry logic.

    This handles:
        - 429 Too Many Requests
        - request timeouts
        - temporary connection errors

    Retry behavior:
        - Wait longer after each failed attempt.
        - Try the same request again.
        - Raise the error only after repeated failures.
    """

    if retry_waits is None:
        retry_waits = [30, 60, 120, 180, 240, 300]

    url = f"{BIRDEYE_BASE_URL}{endpoint}"

    headers = {
        "accept": "application/json",
        "X-API-KEY": BIRDEYE_API_KEY,
        "x-chain": "solana",
    }

    attempt = 0

    while True:
        try:
            wait_for_request_slot()

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=90,
            )

            mark_request_finished()

            if response.ok:
                return response.json()

            # Handle rate limits.
            if response.status_code == 429 and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")

                if retry_after is not None:
                    try:
                        wait_seconds = int(float(retry_after))
                    except ValueError:
                        wait_seconds = retry_waits[min(attempt, len(retry_waits) - 1)]
                else:
                    wait_seconds = retry_waits[min(attempt, len(retry_waits) - 1)]

                print(
                    f"Rate limit hit for {endpoint}. "
                    f"Waiting {wait_seconds} seconds before retry {attempt + 1}..."
                )

                time.sleep(wait_seconds)
                attempt += 1
                continue

            print("\nBirdeye request failed.")
            print(f"Endpoint: {endpoint}")
            print(f"Params: {params}")
            print(f"Status code: {response.status_code}")
            print(f"Response body: {response.text}")

            response.raise_for_status()

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as error:
            mark_request_finished()

            if attempt < max_retries:
                wait_seconds = retry_waits[min(attempt, len(retry_waits) - 1)]

                print(
                    f"Network/timeout error for {endpoint}: {error}. "
                    f"Waiting {wait_seconds} seconds before retry {attempt + 1}..."
                )

                time.sleep(wait_seconds)
                attempt += 1
                continue

            print("\nBirdeye request failed after repeated network/timeout errors.")
            print(f"Endpoint: {endpoint}")
            print(f"Params: {params}")
            raise


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def safe_float(value):
    """
    Convert a value to float when possible.

    Returns None when the value is missing or invalid.
    """

    try:
        if value is None:
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def first_existing_value(data: dict, keys: list[str], default=None):
    """
    Return the first non-empty value from a list of possible dictionary keys.
    """

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


# ---------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------

def get_token_price_usd(token_address: str) -> float | None:
    """
    Get the latest USD price for a token.
    """

    response_json = birdeye_get(
        endpoint="/defi/price",
        params={"address": token_address},
    )

    data = response_json.get("data", {})

    price = first_existing_value(
        data,
        ["value", "price", "priceUsd", "price_usd"],
    )

    return safe_float(price)


def get_sol_price_usd() -> float | None:
    """
    Get the current SOL/USD price using wrapped SOL.
    """

    return get_token_price_usd(WRAPPED_SOL_ADDRESS)


def get_token_overview(token_address: str) -> dict:
    """
    Get overview data for one token.
    """

    response_json = birdeye_get(
        endpoint="/defi/token_overview",
        params={
            "address": token_address,
            "frames": "1m,5m,30m,1h,8h,24h",
        },
    )

    return response_json.get("data", {})


def extract_holder_items(response_json: dict) -> list[dict]:
    """
    Extract holder rows from the holder endpoint response.
    """

    data = response_json.get("data")

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]

    if isinstance(data, dict) and isinstance(data.get("holders"), list):
        return data["holders"]

    if isinstance(data, dict) and isinstance(data.get("list"), list):
        return data["list"]

    return []


def get_top_holders(token_address: str, limit: int = 100) -> list[dict]:
    """
    Get top holder rows for one token.
    """

    response_json = birdeye_get(
        endpoint="/defi/v3/token/holder",
        params={
            "address": token_address,
            "offset": 0,
            "limit": limit,
            "ui_amount_mode": "scaled",
        },
    )

    return extract_holder_items(response_json)