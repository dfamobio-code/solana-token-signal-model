"""
collect_candidate_pool.py

Collect a candidate pool of newly listed Solana tokens from Birdeye.

Purpose:
    Pull up to 1000 tokens listed within the last 8 hours.

Important:
    This script only builds the candidate pool.

    It does not calculate WDS.
    It does not pull top-holder data.
    It does not select the 200 tracking tokens.

Research flow:
    1. Collect candidate pool of up to 1000 newly listed tokens.
    2. Randomly sample 200 tokens using a fixed seed.
    3. Collect full initial features and WDS for only those 200 tokens.
    4. Collect follow-up outcome data about 24 hours later.
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from birdeye_client import birdeye_get, first_existing_value, safe_float


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# Path to this file:
# Example:
#   project_root/src/collect_candidate_pool.py
CURRENT_FILE = Path(__file__).resolve()

# Project root:
# Example:
#   project_root/
PROJECT_ROOT = CURRENT_FILE.parents[1]

# Raw data output folder:
# Example:
#   project_root/data/raw/
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Partial output folder:
# Example:
#   project_root/data/raw/partial/
PARTIAL_DATA_DIR = RAW_DATA_DIR / "partial"

# Create output folders if they do not already exist.
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------

def extract_new_listing_items(response_json: dict) -> list[dict]:
    """
    Extract token rows from the Birdeye new-listing response.

    API responses are sometimes nested differently, so this function checks
    multiple possible response shapes.

    Possible shapes:
        response["data"]
        response["data"]["items"]
        response["data"]["tokens"]
        response["data"]["list"]

    Returns
    -------
    list[dict]:
        List of token dictionaries.
    """

    data = response_json.get("data")

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]

    if isinstance(data, dict) and isinstance(data.get("tokens"), list):
        return data["tokens"]

    if isinstance(data, dict) and isinstance(data.get("list"), list):
        return data["list"]

    return []


def unix_to_utc_string(unix_time) -> str | None:
    """
    Convert a Unix timestamp to a readable UTC datetime string.
    """

    unix_time = safe_float(unix_time)

    if unix_time is None:
        return None

    try:
        return datetime.fromtimestamp(unix_time, tz=timezone.utc).isoformat()
    except (OSError, ValueError):
        return None


def save_partial_candidate_pool(rows: list[dict], run_id: str) -> Path:
    """
    Save the currently collected candidate rows to a partial CSV.

    This protects against losing all progress if the script is interrupted,
    rate-limited too heavily, or crashes before the final save.

    The same partial file is overwritten each time.
    """

    partial_path = PARTIAL_DATA_DIR / f"candidate_pool_partial_{run_id}.csv"

    partial_df = pd.DataFrame(rows)

    partial_df.to_csv(partial_path, index=False)

    return partial_path


# ---------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------

def collect_candidate_pool(
    max_candidates: int,
    max_age_hours: int,
    page_sleep_seconds: float,
    partial_save_every_pages: int,
) -> Path:
    """
    Collect newly listed token candidates from Birdeye.

    Parameters
    ----------
    max_candidates:
        Maximum number of candidate tokens to collect.
        Example:
            1000

    max_age_hours:
        Maximum listing age in hours.
        Example:
            8

    page_sleep_seconds:
        Delay between successful new-listing page requests.
        This helps avoid rate-limit errors.

    partial_save_every_pages:
        Save a partial CSV after this many successful pages.

    Returns
    -------
    Path:
        Path to the final saved candidate pool CSV.
    """

    # This ID is used so all files from the same run share the same timestamp.
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

    # Current Unix time in seconds.
    now_unix = int(time.time())

    # Oldest acceptable listing time.
    #
    # Example:
    #   If max_age_hours = 8, tokens listed more than 8 hours ago are skipped.
    cutoff_unix = now_unix - max_age_hours * 60 * 60

    # time_to is used to walk backward through the new-listing endpoint.
    time_to = now_unix

    # Collected token rows.
    rows = []

    # Used to prevent duplicate token addresses.
    seen_addresses = set()

    # Page counter for progress tracking and partial saves.
    pages_requested = 0
    successful_pages = 0

    print(
        f"Collecting up to {max_candidates} tokens "
        f"listed within the last {max_age_hours} hours..."
    )

    print(f"Run ID: {run_id}")
    print(f"Cutoff UTC: {unix_to_utc_string(cutoff_unix)}")
    print(f"Page sleep seconds: {page_sleep_seconds}")
    print()

    while len(rows) < max_candidates:
        pages_requested += 1

        print(
            f"Requesting page {pages_requested} "
            f"with time_to={time_to} ({unix_to_utc_string(time_to)})..."
        )

        try:
            response_json = birdeye_get(
                endpoint="/defi/v2/tokens/new_listing",
                params={
                    # Birdeye's new listing endpoint has a small page size.
                    "limit": 20,

                    # Walk backward through listing history.
                    "time_to": time_to,

                    # Solana meme-platform listings.
                    "meme_platform_enabled": "true",
                },
            )

        except requests.HTTPError as error:
            print(f"HTTP error while requesting new listings: {error}")

            # Save whatever has been collected so far before stopping.
            if rows:
                partial_path = save_partial_candidate_pool(rows, run_id)
                print(f"Saved partial candidate pool to: {partial_path}")

            raise

        tokens = extract_new_listing_items(response_json)

        if not tokens:
            print("No tokens returned by the new-listing endpoint.")
            break

        successful_pages += 1

        oldest_listing_time_in_page = None
        added_from_page = 0
        skipped_duplicates = 0
        skipped_too_old = 0
        skipped_missing_address = 0

        for token in tokens:
            # Try several possible address field names.
            address = first_existing_value(
                token,
                ["address", "token_address", "mint", "mint_address"],
            )

            # Try several possible listing-time field names.
            listing_time = first_existing_value(
                token,
                [
                    "listing_time",
                    "listed_at",
                    "recent_listing_time",
                    "created_at",
                    "launch_time",
                ],
            )

            listing_time_float = safe_float(listing_time)

            if not address:
                skipped_missing_address += 1
                continue

            if address in seen_addresses:
                skipped_duplicates += 1
                continue

            if listing_time_float is not None and listing_time_float < cutoff_unix:
                skipped_too_old += 1
                continue

            if listing_time_float is not None:
                if oldest_listing_time_in_page is None:
                    oldest_listing_time_in_page = listing_time_float
                else:
                    oldest_listing_time_in_page = min(
                        oldest_listing_time_in_page,
                        listing_time_float,
                    )

            collected_at_unix = int(time.time())

            if listing_time_float is not None:
                age_minutes_at_pull = (collected_at_unix - listing_time_float) / 60
            else:
                age_minutes_at_pull = None

            rows.append(
                {
                    # Token identity
                    "address": address,
                    "name": first_existing_value(token, ["name", "token_name"]),
                    "symbol": first_existing_value(token, ["symbol", "token_symbol"]),

                    # Listing metadata
                    "listing_time_unix": listing_time_float,
                    "listing_time_utc": unix_to_utc_string(listing_time_float),
                    "age_minutes_at_candidate_pull": age_minutes_at_pull,

                    # Candidate collection metadata
                    "candidate_collected_at_unix": collected_at_unix,
                    "candidate_collected_at_utc": datetime.fromtimestamp(
                        collected_at_unix,
                        tz=timezone.utc,
                    ).isoformat(),
                    "candidate_run_id": run_id,
                    "candidate_page_number": successful_pages,

                    # Raw fields that may be useful if Birdeye provides them
                    "logo_uri": first_existing_value(token, ["logo_uri", "logoURI"]),
                    "decimals": first_existing_value(token, ["decimals"]),
                }
            )

            seen_addresses.add(address)
            added_from_page += 1

            if len(rows) >= max_candidates:
                break

        # Move backward in time for the next page.
        if oldest_listing_time_in_page is not None:
            time_to = int(oldest_listing_time_in_page) - 1
        else:
            # Fallback if the response does not include listing times.
            # This prevents requesting the same time window forever.
            time_to -= 60

        print(
            f"Page {successful_pages} complete. "
            f"Added: {added_from_page}. "
            f"Total collected: {len(rows)}/{max_candidates}. "
            f"Skipped duplicates: {skipped_duplicates}. "
            f"Skipped too old: {skipped_too_old}. "
            f"Skipped missing address: {skipped_missing_address}."
        )

        # Save partial progress every few successful pages.
        if successful_pages % partial_save_every_pages == 0:
            partial_path = save_partial_candidate_pool(rows, run_id)
            print(f"Partial save written to: {partial_path}")

        print()

        # Stop if the next request would be older than the cutoff.
        if time_to < cutoff_unix:
            print("Reached the max-age cutoff.")
            break

        # Slow down between successful page requests to reduce rate-limit risk.
        time.sleep(page_sleep_seconds)

    # Final save.
    final_df = pd.DataFrame(rows)

    output_path = RAW_DATA_DIR / f"candidate_pool_{run_id}.csv"

    final_df.to_csv(output_path, index=False)

    # Also save one final partial file for safety.
    partial_path = save_partial_candidate_pool(rows, run_id)

    print()
    print(f"Saved final candidate pool to: {output_path}")
    print(f"Saved final partial backup to: {partial_path}")
    print(f"Final candidate count: {len(final_df)}")

    if len(final_df) < max_candidates:
        print(
            "Warning: fewer candidates were collected than requested. "
            "This can happen if fewer tokens were available within the time window "
            "or if the endpoint stopped returning usable results."
        )

    return output_path


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    Example from project root:
        python3 src/collect_candidate_pool.py \
            --max-candidates 1000 \
            --max-age-hours 8

    Example from inside src:
        python3 collect_candidate_pool.py \
            --max-candidates 1000 \
            --max-age-hours 8
    """

    parser = argparse.ArgumentParser(
        description="Collect newly listed Solana token candidates from Birdeye."
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1000,
        help="Maximum number of candidate tokens to collect.",
    )

    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=8,
        help="Maximum token listing age in hours.",
    )

    parser.add_argument(
        "--page-sleep-seconds",
        type=float,
        default=2.0,
        help="Pause between successful new-listing page requests.",
    )

    parser.add_argument(
        "--partial-save-every-pages",
        type=int,
        default=5,
        help="Save partial progress after this many successful pages.",
    )

    return parser.parse_args()


def main():
    """
    Main entry point.
    """

    args = parse_args()

    collect_candidate_pool(
        max_candidates=args.max_candidates,
        max_age_hours=args.max_age_hours,
        page_sleep_seconds=args.page_sleep_seconds,
        partial_save_every_pages=args.partial_save_every_pages,
    )


if __name__ == "__main__":
    main()