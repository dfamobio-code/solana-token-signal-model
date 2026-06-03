"""
test_wds_holder_endpoint.py

This script tests whether the Birdeye holder endpoint gives enough data
to calculate Whale Dominance Score (WDS).

The goal is not to build the full dataset yet.

The goal is only to answer these questions:

1. Does the endpoint work with the current API key?
2. Does it return the top holders for a token?
3. Does it return holder balances or ownership percentages?
4. Can WDS be calculated from the returned holder data?
5. What column names does Birdeye actually return?

Run example:

    python src/test_wds_holder_endpoint.py --address So11111111111111111111111111111111111111112

A better test is to use a real memecoin mint address instead of wrapped SOL.
"""

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------

# Load variables from the .env file.
# This expects a .env file in the project root with:
#
#     BIRDEYE_API_KEY=the_api_key
#
load_dotenv()

# Read the Birdeye API key from the environment.
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")

# Stop immediately if the API key is missing.
if not BIRDEYE_API_KEY:
    raise ValueError(
        "BIRDEYE_API_KEY was not found. "
        "Add it to the .env file before running this script."
    )


# Birdeye API base URL.
BIRDEYE_BASE_URL = "https://public-api.birdeye.so"


# ---------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------

def birdeye_get(endpoint: str, params: dict) -> dict:
    """
    Send a GET request to Birdeye.

    This helper exists so request headers and error handling are kept
    in one place.

    Parameters
    ----------
    endpoint:
        Birdeye endpoint path.
        Example:
            "/defi/v3/token/holder"

    params:
        Query parameters for the request.
        Example:
            {
                "address": "...",
                "offset": 0,
                "limit": 100,
                "ui_amount_mode": "scaled"
            }

    Returns
    -------
    dict:
        Parsed JSON response.
    """

    # Build the full request URL.
    url = f"{BIRDEYE_BASE_URL}{endpoint}"

    # Required headers.
    # x-chain tells Birdeye this request is for Solana.
    headers = {
        "accept": "application/json",
        "X-API-KEY": BIRDEYE_API_KEY,
        "x-chain": "solana",
    }

    # Print the request so it is clear what is being tested.
    print("\nSending request:")
    print(f"URL: {url}")
    print(f"Params: {params}")

    # Send the request.
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    # Print status code before raising errors.
    # This makes debugging easier if the endpoint is blocked by plan/rate limits.
    print(f"\nHTTP status code: {response.status_code}")

    # If the request failed, print the response body before raising.
    if not response.ok:
        print("\nRequest failed. Response body:")
        print(response.text)
        response.raise_for_status()

    # Return parsed JSON.
    return response.json()


# ---------------------------------------------------------------------
# Holder extraction helpers
# ---------------------------------------------------------------------

def extract_holder_items(response_json: dict) -> list[dict]:
    """
    Extract holder rows from the Birdeye response.

    API responses can be nested in different ways, so this function checks
    multiple common structures.

    Possible response shapes include:

        response["data"]["items"]
        response["data"]["holders"]
        response["data"]

    Returns
    -------
    list[dict]:
        Holder records.
    """

    data = response_json.get("data")

    # Case 1:
    # data is already a list of holders.
    if isinstance(data, list):
        return data

    # Case 2:
    # data is a dictionary containing an items list.
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]

    # Case 3:
    # data is a dictionary containing a holders list.
    if isinstance(data, dict) and isinstance(data.get("holders"), list):
        return data["holders"]

    # Case 4:
    # data is a dictionary containing a list under another common name.
    if isinstance(data, dict) and isinstance(data.get("list"), list):
        return data["list"]

    # No known holder list found.
    return []


def first_existing_value(row: dict, possible_keys: list[str]):
    """
    Return the first non-empty value found from a list of possible keys.

    This is useful because APIs may use different names for the same idea.

    Example:
        holder address might be:
            "address"
            "owner"
            "wallet"
            "wallet_address"
    """

    for key in possible_keys:
        if key in row and row[key] is not None:
            return row[key]

    return None


def safe_float(value):
    """
    Convert a value to float when possible.

    Returns None if the value is missing or cannot be converted.
    """

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# WDS calculation
# ---------------------------------------------------------------------

def calculate_wds_from_shares(shares: list[float]) -> dict:
    """
    Calculate WDS using top-holder ownership shares.

    The paper defines:

        C_i = sum of top n holder shares

        H_i = sum of squared top n holder shares

        N_i = ((H_i / C_i^2) - (1/n)) / (1 - (1/n))

        WDS_i = C_i * N_i

    Important:
        The shares must be fractions, not percentages.

    Example:
        5% should be represented as 0.05, not 5.

    Parameters
    ----------
    shares:
        Ownership shares for the top holders.
        Example:
            [0.20, 0.10, 0.05, ...]

    Returns
    -------
    dict:
        WDS components.
    """

    # Remove missing values and keep only valid non-negative numbers.
    cleaned_shares = []

    for share in shares:
        share = safe_float(share)

        if share is None:
            continue

        if share < 0:
            continue

        cleaned_shares.append(share)

    # n is the number of holder shares actually used.
    n = len(cleaned_shares)

    # WDS cannot be calculated without at least 2 holders.
    # With only 1 holder, the normalization denominator becomes awkward
    # for this concentration comparison.
    if n < 2:
        return {
            "can_calculate_wds": False,
            "reason": "Need at least 2 holder shares.",
            "n_used": n,
            "C": None,
            "H": None,
            "N": None,
            "WDS": None,
        }

    # C_i: cumulative share of the top n holders.
    C = sum(cleaned_shares)

    # If C is zero, H / C^2 would divide by zero.
    if C == 0:
        return {
            "can_calculate_wds": False,
            "reason": "Cumulative holder share is zero.",
            "n_used": n,
            "C": C,
            "H": None,
            "N": None,
            "WDS": None,
        }

    # H_i: Herfindahl-style concentration among the top holders.
    H = sum(share ** 2 for share in cleaned_shares)

    # Normalized internal concentration.
    N = ((H / (C ** 2)) - (1 / n)) / (1 - (1 / n))

    # Whale Dominance Score.
    WDS = C * N

    return {
        "can_calculate_wds": True,
        "reason": None,
        "n_used": n,
        "C": C,
        "H": H,
        "N": N,
        "WDS": WDS,
    }


def extract_shares_from_holders(holder_rows: list[dict]) -> list[float]:
    """
    Try to extract ownership shares directly from holder rows.

    Some APIs return ownership as a percentage or share.

    This function checks common possible field names.

    Returns
    -------
    list[float]:
        Holder shares as fractions.

    Notes
    -----
    If the API returns 5 for 5%, this converts it to 0.05.

    If the API returns 0.05 for 5%, this keeps it as 0.05.
    """

    shares = []

    for row in holder_rows:
        # Try several possible names for holder ownership percentage/share.
        raw_share = first_existing_value(
            row,
            [
                "share",
                "percentage",
                "percent",
                "pct",
                "ownership_share",
                "ownership_percentage",
                "ui_amount_percent",
            ],
        )

        value = safe_float(raw_share)

        if value is None:
            continue

        # If the value is greater than 1, assume it is a percentage.
        # Example:
        #     5 means 5%, so convert to 0.05.
        #
        # If the value is already 0.05, leave it alone.
        if value > 1:
            value = value / 100

        shares.append(value)

    return shares


def extract_balances_from_holders(holder_rows: list[dict]) -> list[float]:
    """
    Try to extract holder balances from holder rows.

    This is useful if the API does not directly return ownership shares.

    Returns
    -------
    list[float]:
        Holder balances.
    """

    balances = []

    for row in holder_rows:
        raw_balance = first_existing_value(
            row,
            [
                "ui_amount",
                "amount",
                "balance",
                "token_balance",
                "quantity",
            ],
        )

        value = safe_float(raw_balance)

        if value is None:
            continue

        balances.append(value)

    return balances


def convert_balances_to_shares(balances: list[float], supply: float) -> list[float]:
    """
    Convert raw holder balances into ownership shares.

    Formula:
        ownership_share = holder_balance / supply

    Parameters
    ----------
    balances:
        Holder token balances.

    supply:
        Total supply or circulating supply.

    Returns
    -------
    list[float]:
        Ownership shares as fractions.
    """

    supply = safe_float(supply)

    if supply is None or supply <= 0:
        return []

    shares = []

    for balance in balances:
        balance = safe_float(balance)

        if balance is None:
            continue

        shares.append(balance / supply)

    return shares


# ---------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------

def test_holder_endpoint(token_address: str, supply: float | None, save_json: bool) -> None:
    """
    Test Birdeye's holder endpoint for one token.

    Parameters
    ----------
    token_address:
        Token mint address to test.

    supply:
        Optional token supply.
        This is only needed if Birdeye returns balances but not shares.

    save_json:
        Whether to save the raw response to a JSON file.
    """

    # Call Birdeye's top-holder endpoint.
    response_json = birdeye_get(
        endpoint="/defi/v3/token/holder",
        params={
            "address": token_address,
            "offset": 0,
            "limit": 100,
            "ui_amount_mode": "scaled",
        },
    )

    # Optionally save the full raw API response.
    # This is useful for inspecting the exact response shape.
    if save_json:
        output_dir = Path("data/raw/api_tests")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"holder_test_{token_address}.json"

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(response_json, file, indent=2)

        print(f"\nSaved raw response to: {output_path}")

    # Extract holder rows from the response.
    holder_rows = extract_holder_items(response_json)

    print(f"\nNumber of holder rows returned: {len(holder_rows)}")

    # If no holder rows are returned, print the top-level response keys
    # to help inspect the structure.
    if not holder_rows:
        print("\nNo holder rows found using the current extraction logic.")
        print("Top-level response keys:")
        print(list(response_json.keys()))

        print("\nFull response preview:")
        print(json.dumps(response_json, indent=2)[:3000])
        return

    # Print the keys from the first holder row.
    # This is the most important part of the test.
    print("\nKeys in the first holder row:")
    print(list(holder_rows[0].keys()))

    # Print the first few holder rows.
    print("\nFirst 3 holder rows:")
    print(json.dumps(holder_rows[:3], indent=2)[:3000])

    # First, try to calculate WDS using direct ownership share fields.
    shares = extract_shares_from_holders(holder_rows)

    if shares:
        print(f"\nFound {len(shares)} ownership share values directly from holder rows.")

        wds_result = calculate_wds_from_shares(shares)

        print("\nWDS result using direct shares:")
        print(json.dumps(wds_result, indent=2))

        return

    # If no shares were found, try to extract balances.
    balances = extract_balances_from_holders(holder_rows)

    print(f"\nFound {len(balances)} holder balance values.")

    # If balances exist but no supply was provided, explain what is missing.
    if balances and supply is None:
        print("\nBalances were found, but no supply was provided.")
        print("To calculate WDS from balances, run again with:")
        print("    --supply SOME_SUPPLY_VALUE")
        print("\nExample:")
        print(f"    python src/test_wds_holder_endpoint.py --address {token_address} --supply 1000000000")
        return

    # Convert balances to shares if supply was provided.
    if balances and supply is not None:
        shares = convert_balances_to_shares(balances, supply)

        print(f"\nConverted {len(shares)} balances into ownership shares.")

        wds_result = calculate_wds_from_shares(shares)

        print("\nWDS result using balances / supply:")
        print(json.dumps(wds_result, indent=2))

        return

    # If neither shares nor balances were found, the endpoint response
    # does not currently contain enough data using the known field names.
    print("\nCould not find ownership shares or balances in the holder rows.")
    print("Inspect the printed holder row keys and update the field names in the script.")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    Example with only token address:

        python src/test_wds_holder_endpoint.py \
            --address So11111111111111111111111111111111111111112

    Example with supply:

        python src/test_wds_holder_endpoint.py \
            --address SOME_TOKEN_ADDRESS \
            --supply 1000000000

    Example saving raw JSON:

        python src/test_wds_holder_endpoint.py \
            --address SOME_TOKEN_ADDRESS \
            --save-json
    """

    parser = argparse.ArgumentParser(
        description="Test whether Birdeye holder data can support WDS calculation."
    )

    parser.add_argument(
        "--address",
        required=True,
        help="Solana token mint address to test.",
    )

    parser.add_argument(
        "--supply",
        type=float,
        default=None,
        help="Optional token supply used to convert holder balances into shares.",
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save the raw Birdeye response to data/raw/api_tests.",
    )

    return parser.parse_args()


def main():
    """
    Main entry point.
    """

    args = parse_args()

    test_holder_endpoint(
        token_address=args.address,
        supply=args.supply,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()