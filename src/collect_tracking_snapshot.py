"""
collect_tracking_snapshot.py

Collect the full initial feature snapshot for selected tracking tokens.

Purpose:
    For the selected 200 tokens:
        - pull overview data
        - pull price data
        - pull SOL/USD reference price
        - pull top-holder balances
        - calculate WDS
        - save the initial tracking snapshot

This file creates the model-input dataset.
The follow-up script later creates the outcome dataset.
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
    get_top_holders,
    safe_float,
    set_request_interval,
)
from wds import extract_holder_balances, calculate_wds_from_balances


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

def choose_supply(overview: dict, tracking_row: pd.Series) -> float | None:
    """
    Choose supply for WDS.

    Preference order:
        1. circulating_supply from overview
        2. total_supply from overview
        3. circulating_supply from tracking CSV
        4. total_supply from tracking CSV
    """

    candidates = [
        first_existing_value(overview, ["circulating_supply", "circulatingSupply"]),
        first_existing_value(overview, ["total_supply", "totalSupply"]),
        tracking_row.get("circulating_supply"),
        tracking_row.get("total_supply"),
    ]

    for value in candidates:
        value = safe_float(value)

        if value is not None and value > 0:
            return value

    return None


def derive_price_sol(price_usd: float | None, sol_price_usd: float | None) -> float | None:
    """
    Convert USD token price into SOL terms.
    """

    price_usd = safe_float(price_usd)
    sol_price_usd = safe_float(sol_price_usd)

    if price_usd is None:
        return None

    if sol_price_usd is None or sol_price_usd <= 0:
        return None

    return price_usd / sol_price_usd


def save_partial_snapshot(rows: list[dict], run_id: str) -> Path:
    """
    Save collected rows to a partial CSV.

    The same partial file is overwritten after every token.
    This prevents losing progress if the script is interrupted.
    """

    partial_path = PARTIAL_DATA_DIR / f"initial_tracking_snapshot_partial_{run_id}.csv"

    partial_df = pd.DataFrame(rows)
    partial_df.to_csv(partial_path, index=False)

    return partial_path


def load_existing_partial(run_id: str) -> tuple[list[dict], set[str]]:
    """
    Load an existing partial snapshot if one exists.

    This supports resuming the same run after an interruption.
    """

    partial_path = PARTIAL_DATA_DIR / f"initial_tracking_snapshot_partial_{run_id}.csv"

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

    print(f"Loaded existing partial snapshot: {partial_path}")
    print(f"Completed tokens loaded: {len(completed_addresses)}")

    return rows, completed_addresses


def build_initial_snapshot_row(
    tracking_row: pd.Series,
    sol_price_usd: float | None,
) -> dict:
    """
    Build one full initial snapshot row for one selected token.
    """

    address = str(tracking_row["address"])

    overview = get_token_overview(address)

    price_usd = get_token_price_usd(address)
    price_sol = derive_price_sol(price_usd, sol_price_usd)

    supply_for_wds = choose_supply(overview, tracking_row)

    holder_rows = get_top_holders(address, limit=100)
    holder_balances = extract_holder_balances(holder_rows)

    wds_result = calculate_wds_from_balances(
        balances=holder_balances,
        supply=supply_for_wds,
    )

    collected_at_unix = int(time.time())
    collected_at_utc = datetime.fromtimestamp(
        collected_at_unix,
        tz=timezone.utc,
    ).isoformat()

    return {
        # Token identity
        "address": address,
        "name": tracking_row.get("name"),
        "symbol": tracking_row.get("symbol"),

        # Snapshot metadata
        "initial_collected_at_unix": collected_at_unix,
        "initial_collected_at_utc": collected_at_utc,

        # Sampling metadata
        "random_seed": tracking_row.get("random_seed"),
        "sample_size": tracking_row.get("sample_size"),
        "sampled_at_utc": tracking_row.get("sampled_at_utc"),

        # Listing metadata
        "listing_time_unix": tracking_row.get("listing_time_unix"),
        "listing_time_utc": tracking_row.get("listing_time_utc"),
        "age_minutes_at_candidate_pull": tracking_row.get("age_minutes_at_candidate_pull"),

        # Required model features
        "circulating_supply": first_existing_value(
            overview,
            ["circulating_supply", "circulatingSupply"],
            default=tracking_row.get("circulating_supply"),
        ),
        "total_supply": first_existing_value(
            overview,
            ["total_supply", "totalSupply"],
            default=tracking_row.get("total_supply"),
        ),
        "liquidity": first_existing_value(
            overview,
            ["liquidity", "liquidity_usd"],
        ),
        "price_usd_initial": price_usd,
        "price_sol_initial": price_sol,
        "holders": first_existing_value(
            overview,
            ["holder", "holders", "holder_count", "number_holders"],
        ),
        "fdv": first_existing_value(
            overview,
            ["fdv", "fully_diluted_valuation"],
        ),
        "txns": first_existing_value(
            overview,
            [
                "trade_8h_count",
                "txns_8h",
                "transaction_8h_count",
                "trade_24h_count",
                "txns_24h",
                "transaction_24h_count",
                "trade_count",
                "txns",
            ],
        ),

        # WDS fields
        "supply_used_for_wds": supply_for_wds,
        "holder_rows_returned": len(holder_rows),
        "holder_balances_extracted": len(holder_balances),
        **wds_result,

        # SOL reference
        "sol_price_usd_initial": sol_price_usd,

        # Extra analysis fields
        "market_cap": first_existing_value(
            overview,
            ["market_cap", "marketCap"],
        ),
        "volume_8h_usd": first_existing_value(
            overview,
            ["volume_8h_usd", "volume8h", "v8hUSD"],
        ),
        "volume_24h_usd_initial": first_existing_value(
            overview,
            ["volume_24h_usd", "volume24h", "v24hUSD"],
        ),
        "trade_24h_count_initial": first_existing_value(
            overview,
            ["trade_24h_count", "txns_24h", "transaction_24h_count"],
        ),
        "buy_24h_initial": first_existing_value(
            overview,
            ["buy_24h", "buy24h"],
        ),
        "sell_24h_initial": first_existing_value(
            overview,
            ["sell_24h", "sell24h"],
        ),
        "unique_wallet_24h_initial": first_existing_value(
            overview,
            ["unique_wallet_24h", "uniqueWallet24h"],
        ),

        # Error field kept empty for successful rows.
        "snapshot_error": None,
    }


# ---------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------

def collect_tracking_snapshot(
    tracking_csv: Path,
    token_sleep_seconds: float,
    request_sleep_seconds: float,
    run_id: str | None,
    resume: bool,
) -> Path:
    """
    Collect initial full feature snapshot for selected tracking tokens.

    Parameters
    ----------
    tracking_csv:
        CSV containing selected token addresses.

    token_sleep_seconds:
        Delay after each token finishes.

    request_sleep_seconds:
        Minimum delay between every Birdeye API request.

    run_id:
        Optional run ID used for output files.

    resume:
        Whether to load an existing partial file for the same run ID.
    """

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

    # Apply global request pacing.
    set_request_interval(request_sleep_seconds)

    tracking_df = pd.read_csv(tracking_csv)

    if "address" not in tracking_df.columns:
        raise ValueError("Tracking CSV must contain an 'address' column.")

    tracking_df = tracking_df.dropna(subset=["address"])
    tracking_df = tracking_df.drop_duplicates(subset=["address"])

    print(f"Collecting initial snapshot for {len(tracking_df)} tokens...")
    print(f"Run ID: {run_id}")
    print(f"Request sleep seconds: {request_sleep_seconds}")
    print(f"Token sleep seconds: {token_sleep_seconds}")

    if resume:
        rows, completed_addresses = load_existing_partial(run_id)
    else:
        rows, completed_addresses = [], set()

    # Get SOL/USD once.
    # This also uses the same global request pacing.
    sol_price_usd = get_sol_price_usd()

    print(f"SOL/USD at initial snapshot: {sol_price_usd}")

    for index, (_, tracking_row) in enumerate(tracking_df.iterrows(), start=1):
        address = str(tracking_row["address"])
        symbol = tracking_row.get("symbol")

        if address in completed_addresses:
            print(f"\n[{index}/{len(tracking_df)}] Skipping already completed {symbol} - {address}")
            continue

        print(f"\n[{index}/{len(tracking_df)}] Collecting {symbol} - {address}")

        try:
            row = build_initial_snapshot_row(
                tracking_row=tracking_row,
                sol_price_usd=sol_price_usd,
            )

            rows.append(row)
            completed_addresses.add(address)

            print(f"WDS: {row.get('wds')}")
            print(f"Holder rows returned: {row.get('holder_rows_returned')}")

        except requests.HTTPError as error:
            error_row = {
                "address": address,
                "name": tracking_row.get("name"),
                "symbol": symbol,
                "initial_collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot_error": f"HTTP error: {error}",
            }

            rows.append(error_row)
            completed_addresses.add(address)

            print(f"HTTP error for {address}: {error}")

        except Exception as error:
            error_row = {
                "address": address,
                "name": tracking_row.get("name"),
                "symbol": symbol,
                "initial_collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot_error": f"Unexpected error: {error}",
            }

            rows.append(error_row)
            completed_addresses.add(address)

            print(f"Unexpected error for {address}: {error}")

        partial_path = save_partial_snapshot(rows, run_id)
        print(f"Partial snapshot saved to: {partial_path}")

        time.sleep(token_sleep_seconds)

    output_df = pd.DataFrame(rows)

    output_path = RAW_DATA_DIR / f"initial_tracking_snapshot_{run_id}.csv"

    output_df.to_csv(output_path, index=False)

    print(f"\nSaved initial tracking snapshot to: {output_path}")
    print(f"Rows saved: {len(output_df)}")

    if "snapshot_error" in output_df.columns:
        error_count = output_df["snapshot_error"].notna().sum()
        print(f"Rows with errors: {error_count}")

    return output_path


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Collect full initial snapshot for selected tracking tokens."
    )

    parser.add_argument(
        "--tracking-csv",
        type=Path,
        required=True,
        help="Path to the selected 200-token tracking CSV.",
    )

    parser.add_argument(
        "--token-sleep-seconds",
        type=float,
        default=10.0,
        help="Pause after each token finishes.",
    )

    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=5.0,
        help="Minimum pause between every Birdeye API request.",
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
        help="Resume from a partial snapshot with the same run ID.",
    )

    return parser.parse_args()


def main():
    """
    Main entry point.
    """

    args = parse_args()

    collect_tracking_snapshot(
        tracking_csv=args.tracking_csv,
        token_sleep_seconds=args.token_sleep_seconds,
        request_sleep_seconds=args.request_sleep_seconds,
        run_id=args.run_id,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()