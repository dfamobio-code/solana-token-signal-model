"""
batch_test_wds_from_csv.py

This script tests WDS calculation across multiple tokens from an existing CSV.

Purpose:
    This is not the final data-collection pipeline.

    This is a small validation script used before the real project run.

    It checks that:
        1. Token addresses can be read from an existing CSV.
        2. Supply values can be read from the CSV.
        3. Birdeye holder data can be pulled for multiple tokens.
        4. WDS can be calculated automatically.
        5. Results can be saved to a clean output CSV.

Important:
    This should only be run on a small number of tokens, such as 5 or 10.

    The real project should calculate WDS only after the 200-token tracking
    sample has been selected, not for all 1000 candidate tokens.
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# Path to this script:
# Example: project_root/src/batch_test_wds_from_csv.py
CURRENT_FILE = Path(__file__).resolve()

# Project root is one folder above src/
PROJECT_ROOT = CURRENT_FILE.parents[1]

# Output folder for test results.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Create the output folder if it does not already exist.
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------

# Load the .env file from the project root.
# This avoids depending on the folder where the command is run.
load_dotenv(PROJECT_ROOT / ".env")

# Read the Birdeye API key from the environment.
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")

# Stop early if the API key is missing.
if not BIRDEYE_API_KEY:
    raise ValueError(
        "BIRDEYE_API_KEY was not found. "
        "Make sure the .env file exists in the project root."
    )


# ---------------------------------------------------------------------
# Birdeye setup
# ---------------------------------------------------------------------

BIRDEYE_BASE_URL = "https://public-api.birdeye.so"


def birdeye_get(endpoint: str, params: dict) -> dict:
    """
    Send a GET request to the Birdeye API.

    Parameters
    ----------
    endpoint:
        Endpoint path after the base URL.
        Example:
            "/defi/v3/token/holder"

    params:
        Query parameters sent with the request.

    Returns
    -------
    dict:
        Parsed JSON response from Birdeye.
    """

    url = f"{BIRDEYE_BASE_URL}{endpoint}"

    headers = {
        "accept": "application/json",
        "X-API-KEY": BIRDEYE_API_KEY,
        "x-chain": "solana",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    # Print failed response bodies to make debugging easier.
    if not response.ok:
        print("\nRequest failed.")
        print(f"Status code: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status()

    return response.json()


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def safe_float(value):
    """
    Convert a value to float if possible.

    Returns None when the value is missing or invalid.
    """

    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def extract_holder_items(response_json: dict) -> list[dict]:
    """
    Extract holder rows from the Birdeye holder endpoint response.

    The response usually stores holders inside response_json["data"]["items"],
    but this function checks a few possible formats to be safer.
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
    Pull top holder records for one token.

    Parameters
    ----------
    token_address:
        Solana token mint address.

    limit:
        Maximum number of holder records to request.
        For WDS, the target is 100.

    Returns
    -------
    list[dict]:
        Holder rows returned by Birdeye.
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


def extract_holder_balances(holder_rows: list[dict]) -> list[float]:
    """
    Extract holder balances from Birdeye holder rows.

    The key field from the test endpoint was:
        ui_amount

    That value is already adjusted for token decimals.
    """

    balances = []

    for row in holder_rows:
        balance = safe_float(row.get("ui_amount"))

        if balance is None:
            continue

        if balance < 0:
            continue

        balances.append(balance)

    return balances


def calculate_wds_from_balances(
    balances: list[float],
    supply: float,
) -> dict:
    """
    Calculate Whale Dominance Score from holder balances.

    The WDS formula needs ownership shares.

    Step 1:
        ownership_share = holder_balance / supply

    Step 2:
        C = sum of top holder shares

    Step 3:
        H = sum of squared top holder shares

    Step 4:
        N = normalized concentration score

    Step 5:
        WDS = C * N

    Notes:
        The paper uses top 100 holders.

        If fewer than 100 holder rows are available, this function uses all
        available holder rows and records the actual number used.
    """

    supply = safe_float(supply)

    if supply is None or supply <= 0:
        return {
            "can_calculate_wds": False,
            "reason": "Missing or invalid supply.",
            "wds_n_used": 0,
            "wds_c": None,
            "wds_h": None,
            "wds_n": None,
            "wds": None,
        }

    shares = []

    for balance in balances:
        balance = safe_float(balance)

        if balance is None:
            continue

        if balance < 0:
            continue

        shares.append(balance / supply)

    n = len(shares)

    if n < 2:
        return {
            "can_calculate_wds": False,
            "reason": "Need at least 2 holder balances.",
            "wds_n_used": n,
            "wds_c": None,
            "wds_h": None,
            "wds_n": None,
            "wds": None,
        }

    C = sum(shares)

    if C <= 0:
        return {
            "can_calculate_wds": False,
            "reason": "Cumulative holder share is zero.",
            "wds_n_used": n,
            "wds_c": C,
            "wds_h": None,
            "wds_n": None,
            "wds": None,
        }

    H = sum(share ** 2 for share in shares)

    # Normalized concentration score.
    #
    # This rescales concentration so that equal distribution among the
    # selected holders is near 0, while maximum concentration is near 1.
    N = ((H / (C ** 2)) - (1 / n)) / (1 - (1 / n))

    WDS = C * N

    return {
        "can_calculate_wds": True,
        "reason": None,
        "wds_n_used": n,
        "wds_c": C,
        "wds_h": H,
        "wds_n": N,
        "wds": WDS,
    }


def choose_supply(row: pd.Series) -> float | None:
    """
    Choose the supply value used for WDS.

    Preferred:
        circulating_supply

    Fallback:
        total_supply

    Reason:
        Circulating supply is usually closer to the ownership base that is
        actually available in the market. If it is missing, total supply is
        still usable as a fallback for testing.
    """

    circulating_supply = safe_float(row.get("circulating_supply"))

    if circulating_supply is not None and circulating_supply > 0:
        return circulating_supply

    total_supply = safe_float(row.get("total_supply"))

    if total_supply is not None and total_supply > 0:
        return total_supply

    return None


# ---------------------------------------------------------------------
# Main batch test
# ---------------------------------------------------------------------

def run_batch_wds_test(
    input_csv: Path,
    sample_size: int,
    seed: int,
    min_holder_count: int,
    sleep_seconds: float,
) -> Path:
    """
    Run a small batch WDS test using tokens from an existing CSV.

    Parameters
    ----------
    input_csv:
        Existing CSV containing token data.

    sample_size:
        Number of tokens to test.

    seed:
        Random seed used to make the test sample reproducible.

    min_holder_count:
        Optional filter for the CSV holder column.
        Example:
            min_holder_count=50 keeps only tokens where holder >= 50.

    sleep_seconds:
        Pause between API calls to reduce rate-limit risk.

    Returns
    -------
    Path:
        Output CSV path.
    """

    print(f"Reading input CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    if "address" not in df.columns:
        raise ValueError("The input CSV must contain an 'address' column.")

    # Remove rows without token addresses.
    df = df.dropna(subset=["address"])

    # Remove duplicate token addresses.
    df = df.drop_duplicates(subset=["address"])

    # Filter to tokens where supply exists.
    df["chosen_supply_for_wds"] = df.apply(choose_supply, axis=1)
    df = df.dropna(subset=["chosen_supply_for_wds"])

    # If the old CSV has a holder column, use it to avoid testing dead/empty tokens.
    if "holder" in df.columns:
        df["holder"] = pd.to_numeric(df["holder"], errors="coerce")
        df = df[df["holder"].fillna(0) >= min_holder_count]

    if len(df) == 0:
        raise ValueError(
            "No valid tokens left after filtering. "
            "Try lowering --min-holder-count."
        )

    if len(df) < sample_size:
        print(
            f"Only {len(df)} valid tokens are available after filtering. "
            f"Using all of them instead of {sample_size}."
        )
        sample_size = len(df)

    # Randomly sample a small number of tokens.
    sample_df = df.sample(
        n=sample_size,
        random_state=seed,
    ).copy()

    print(f"Testing WDS on {len(sample_df)} tokens...")

    results = []

    for index, (_, row) in enumerate(sample_df.iterrows(), start=1):
        address = str(row["address"])
        symbol = row.get("symbol")
        name = row.get("name")
        csv_holder_count = row.get("holder")
        supply = row.get("chosen_supply_for_wds")

        print(f"\n[{index}/{len(sample_df)}] Testing {symbol} - {address}")
        print(f"Supply used for WDS: {supply}")

        result_row = {
            "address": address,
            "name": name,
            "symbol": symbol,
            "csv_holder_count": csv_holder_count,
            "supply_used_for_wds": supply,
            "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        try:
            holder_rows = get_top_holders(address, limit=100)

            balances = extract_holder_balances(holder_rows)

            wds_result = calculate_wds_from_balances(
                balances=balances,
                supply=supply,
            )

            result_row.update(
                {
                    "holder_rows_returned": len(holder_rows),
                    "holder_balances_extracted": len(balances),
                    **wds_result,
                }
            )

            print(f"Holder rows returned: {len(holder_rows)}")
            print(f"WDS calculated: {wds_result['can_calculate_wds']}")
            print(f"WDS: {wds_result['wds']}")

        except requests.HTTPError as error:
            result_row.update(
                {
                    "holder_rows_returned": None,
                    "holder_balances_extracted": None,
                    "can_calculate_wds": False,
                    "reason": f"HTTP error: {error}",
                    "wds_n_used": None,
                    "wds_c": None,
                    "wds_h": None,
                    "wds_n": None,
                    "wds": None,
                }
            )

            print(f"HTTP error for {address}: {error}")

        except Exception as error:
            result_row.update(
                {
                    "holder_rows_returned": None,
                    "holder_balances_extracted": None,
                    "can_calculate_wds": False,
                    "reason": f"Unexpected error: {error}",
                    "wds_n_used": None,
                    "wds_c": None,
                    "wds_h": None,
                    "wds_n": None,
                    "wds": None,
                }
            )

            print(f"Unexpected error for {address}: {error}")

        results.append(result_row)

        # Pause between requests to reduce rate-limit risk.
        time.sleep(sleep_seconds)

    results_df = pd.DataFrame(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

    output_path = RAW_DATA_DIR / f"batch_wds_test_{timestamp}.csv"

    results_df.to_csv(output_path, index=False)

    print(f"\nSaved batch WDS test results to: {output_path}")

    return output_path


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    Example from the project root:

        python3 src/batch_test_wds_from_csv.py \
            --input-csv data/raw/tokens_discovered.csv \
            --sample-size 10 \
            --seed 20260601 \
            --min-holder-count 20

    Example from inside src:

        python3 batch_test_wds_from_csv.py \
            --input-csv ../data/raw/tokens_discovered.csv \
            --sample-size 10 \
            --seed 20260601 \
            --min-holder-count 20
    """

    parser = argparse.ArgumentParser(
        description="Batch test WDS calculation using token addresses from a CSV."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to the input CSV containing token addresses and supply fields.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of tokens to test.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260601,
        help="Random seed for reproducible sampling.",
    )

    parser.add_argument(
        "--min-holder-count",
        type=int,
        default=20,
        help="Minimum CSV holder count required for a token to be tested.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Pause between API calls.",
    )

    return parser.parse_args()


def main():
    """
    Main entry point.
    """

    args = parse_args()

    run_batch_wds_test(
        input_csv=args.input_csv,
        sample_size=args.sample_size,
        seed=args.seed,
        min_holder_count=args.min_holder_count,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()