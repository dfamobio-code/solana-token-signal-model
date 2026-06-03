"""
collect_followup_snapshot.py

Collect follow-up outcome data for the selected tracking tokens.

Purpose:
    Run this about 24 hours after the initial tracking snapshot.

This script collects outcome data only.

It does not calculate WDS again because WDS is a model-input feature from
the initial time point.

This script is built to be safer for long API runs:
    - global request pacing
    - retry logic through birdeye_client.py
    - partial saving after every token
    - resume support after interruption
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from birdeye_client import (
    first_existing_value,
    get_sol_price_usd,
    get_token_overview,
    get_token_price_usd,
    safe_float,
    set_request_interval,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PARTIAL_DATA_DIR = RAW_DATA_DIR / "partial"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def derive_price_sol(price_usd: float | None, sol_price_usd: float | None) -> float | None:
    """
    Convert a token's USD price into SOL terms.

    Formula:
        token_price_sol = token_price_usd / sol_price_usd

    Returns None if either input is missing or invalid.
    """

    price_usd = safe_float(price_usd)
    sol_price_usd = safe_float(sol_price_usd)

    if price_usd is None:
        return None

    if sol_price_usd is None or sol_price_usd <= 0:
        return None

    return price_usd / sol_price_usd


def save_partial_followup(rows: list[dict], run_id: str) -> Path:
    """
    Save collected follow-up rows to a partial CSV.

    The same partial file is overwritten after every token.

    This means if the script is interrupted, the completed rows are still saved.
    """

    partial_path = PARTIAL_DATA_DIR / f"followup_snapshot_partial_{run_id}.csv"

    partial_df = pd.DataFrame(rows)
    partial_df.to_csv(partial_path, index=False)

    return partial_path


def load_existing_partial(run_id: str) -> tuple[list[dict], set[str]]:
    """
    Load an existing follow-up partial file if it exists.

    This supports resuming the same follow-up run after interruption.

    Returns:
        rows:
            Existing saved rows.

        completed_addresses:
            Token addresses already processed.
    """

    partial_path = PARTIAL_DATA_DIR / f"followup_snapshot_partial_{run_id}.csv"

    if not partial_path.exists():
        return [], set()

    partial_df = pd.read_csv(partial_path)

    if "address" not in partial_df.columns:
        return [], set()

    rows = partial_df.to_dict(orient="records")

    completed_addresses = set(
        partial_df["address"]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(f"Loaded existing partial follow-up snapshot: {partial_path}")
    print(f"Completed tokens loaded: {len(completed_addresses)}")

    return rows, completed_addresses


def build_followup_row(
    token_row: pd.Series,
    sol_price_usd: float | None,
) -> dict:
    """
    Build one follow-up outcome row for one token.

    This function collects only follow-up/outcome fields.

    It intentionally does not:
        - pull top holders
        - calculate WDS
        - create the pumped label

    The pumped label is created later by build_pump_labels.py after joining
    initial and follow-up snapshots.
    """

    address = str(token_row["address"])

    # Pull token overview data.
    overview = get_token_overview(address)

    # Pull current token price in USD.
    price_usd = get_token_price_usd(address)

    # Convert current token price to SOL terms.
    price_sol = derive_price_sol(price_usd, sol_price_usd)

    collected_at_unix = int(time.time())
    collected_at_utc = datetime.fromtimestamp(
        collected_at_unix,
        tz=timezone.utc,
    ).isoformat()

    row = {
        # Token identity
        "address": address,
        "name": token_row.get("name"),
        "symbol": token_row.get("symbol"),

        # Follow-up timestamp
        "followup_collected_at_unix": collected_at_unix,
        "followup_collected_at_utc": collected_at_utc,

        # Final price fields
        "price_usd_followup": price_usd,
        "price_sol_followup": price_sol,
        "sol_price_usd_followup": sol_price_usd,

        # Follow-up market fields
        "liquidity_followup": first_existing_value(
            overview,
            ["liquidity", "liquidity_usd"],
        ),
        "fdv_followup": first_existing_value(
            overview,
            ["fdv", "fully_diluted_valuation"],
        ),
        "market_cap_followup": first_existing_value(
            overview,
            ["market_cap", "marketCap"],
        ),
        "holders_followup": first_existing_value(
            overview,
            ["holder", "holders", "holder_count", "number_holders"],
        ),

        # Follow-up 24h activity fields
        "volume_24h_usd_followup": first_existing_value(
            overview,
            ["volume_24h_usd", "volume24h", "v24hUSD"],
        ),
        "trade_24h_count_followup": first_existing_value(
            overview,
            ["trade_24h_count", "txns_24h", "transaction_24h_count"],
        ),
        "buy_24h_followup": first_existing_value(
            overview,
            ["buy_24h", "buy24h"],
        ),
        "sell_24h_followup": first_existing_value(
            overview,
            ["sell_24h", "sell24h"],
        ),
        "unique_wallet_24h_followup": first_existing_value(
            overview,
            ["unique_wallet_24h", "uniqueWallet24h"],
        ),

        # Extra useful fields if Birdeye provides them
        "price_change_24h_percent_birdeye": first_existing_value(
            overview,
            [
                "price_change_24h_percent",
                "priceChange24hPercent",
                "price_change_24h",
            ],
        ),

        # Error field kept empty for successful rows.
        "followup_error": None,
    }

    return row


# ---------------------------------------------------------------------
# Main follow-up collection function
# ---------------------------------------------------------------------

def collect_followup_snapshot(
    input_csv: Path,
    request_sleep_seconds: float,
    token_sleep_seconds: float,
    run_id: str | None,
    resume: bool,
) -> Path:
    """
    Collect follow-up outcome data for selected tokens.

    Parameters
    ----------
    input_csv:
        CSV containing token addresses.

        This can be either:
            - initial_tracking_snapshot CSV
            - tracking_set CSV

        The preferred input is the initial_tracking_snapshot CSV.

    request_sleep_seconds:
        Minimum delay between every Birdeye API request.

    token_sleep_seconds:
        Extra delay after each token finishes.

    run_id:
        Optional run ID used for output files.

    resume:
        Whether to resume from an existing partial file with the same run ID.
    """

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

    # Apply global request pacing inside birdeye_client.py.
    # This spaces out overview and price calls from one another.
    set_request_interval(request_sleep_seconds)

    token_df = pd.read_csv(input_csv)

    if "address" not in token_df.columns:
        raise ValueError("Input CSV must contain an 'address' column.")

    token_df = token_df.dropna(subset=["address"])
    token_df = token_df.drop_duplicates(subset=["address"])

    print(f"Collecting follow-up snapshot for {len(token_df)} tokens...")
    print(f"Run ID: {run_id}")
    print(f"Request sleep seconds: {request_sleep_seconds}")
    print(f"Token sleep seconds: {token_sleep_seconds}")

    if resume:
        rows, completed_addresses = load_existing_partial(run_id)
    else:
        rows, completed_addresses = [], set()

    # Pull SOL/USD once at the beginning of the follow-up run.
    #
    # This gives a consistent SOL reference price for the follow-up snapshot.
    sol_price_usd = get_sol_price_usd()

    print(f"SOL/USD at follow-up: {sol_price_usd}")

    for index, (_, token_row) in enumerate(token_df.iterrows(), start=1):
        address = str(token_row["address"])
        symbol = token_row.get("symbol")

        if address in completed_addresses:
            print(f"\n[{index}/{len(token_df)}] Skipping already completed {symbol} - {address}")
            continue

        print(f"\n[{index}/{len(token_df)}] Collecting follow-up for {symbol} - {address}")

        try:
            row = build_followup_row(
                token_row=token_row,
                sol_price_usd=sol_price_usd,
            )

            rows.append(row)
            completed_addresses.add(address)

            print(f"Follow-up price USD: {row.get('price_usd_followup')}")
            print(f"Liquidity follow-up: {row.get('liquidity_followup')}")

        except requests.HTTPError as error:
            error_row = {
                "address": address,
                "name": token_row.get("name"),
                "symbol": symbol,
                "followup_collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "followup_error": f"HTTP error: {error}",
            }

            rows.append(error_row)
            completed_addresses.add(address)

            print(f"HTTP error for {address}: {error}")

        except Exception as error:
            error_row = {
                "address": address,
                "name": token_row.get("name"),
                "symbol": symbol,
                "followup_collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "followup_error": f"Unexpected error: {error}",
            }

            rows.append(error_row)
            completed_addresses.add(address)

            print(f"Unexpected error for {address}: {error}")

        # Save progress after every token.
        partial_path = save_partial_followup(rows, run_id)
        print(f"Partial follow-up snapshot saved to: {partial_path}")

        # Extra pause after each token.
        time.sleep(token_sleep_seconds)

    output_df = pd.DataFrame(rows)

    output_path = RAW_DATA_DIR / f"followup_snapshot_{run_id}.csv"

    output_df.to_csv(output_path, index=False)

    print(f"\nSaved follow-up snapshot to: {output_path}")
    print(f"Rows saved: {len(output_df)}")

    if "followup_error" in output_df.columns:
        error_count = output_df["followup_error"].notna().sum()
        print(f"Rows with errors: {error_count}")

    return output_path


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    Example from inside src:

        python3 collect_followup_snapshot.py \
          --input-csv ../data/raw/initial_tracking_snapshot_initial_2026-06-01_201522.csv \
          --request-sleep-seconds 8 \
          --token-sleep-seconds 10 \
          --run-id followup_2026-06-02_201522

    Resume example:

        python3 collect_followup_snapshot.py \
          --input-csv ../data/raw/initial_tracking_snapshot_initial_2026-06-01_201522.csv \
          --request-sleep-seconds 8 \
          --token-sleep-seconds 10 \
          --run-id followup_2026-06-02_201522 \
          --resume
    """

    parser = argparse.ArgumentParser(
        description="Collect 24-hour follow-up outcome snapshot."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Tracking set or initial snapshot CSV containing token addresses.",
    )

    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=8.0,
        help="Minimum pause between every Birdeye API request.",
    )

    parser.add_argument(
        "--token-sleep-seconds",
        type=float,
        default=10.0,
        help="Extra pause after each token finishes.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID. Use the same run ID with --resume to continue a partial run.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a partial follow-up snapshot with the same run ID.",
    )

    return parser.parse_args()


def main():
    """
    Main entry point.
    """

    args = parse_args()

    collect_followup_snapshot(
        input_csv=args.input_csv,
        request_sleep_seconds=args.request_sleep_seconds,
        token_sleep_seconds=args.token_sleep_seconds,
        run_id=args.run_id,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()