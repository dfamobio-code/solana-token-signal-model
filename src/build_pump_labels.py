"""
build_pump_labels.py

Build the final modelling dataset by joining:
    - initial tracking snapshot
    - follow-up snapshot

Target definition:
    pumped = 1 if price_usd increased by more than 50%
    pumped = 0 otherwise

Formula:
    price_change_percent =
        ((price_usd_followup - price_usd_initial) / price_usd_initial) * 100
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_pump_labels(
    initial_csv: Path,
    followup_csv: Path,
    pump_threshold_percent: float,
) -> Path:
    """
    Join initial and follow-up snapshots and create the pump label.
    """

    initial_df = pd.read_csv(initial_csv)
    followup_df = pd.read_csv(followup_csv)

    if "address" not in initial_df.columns:
        raise ValueError("Initial CSV must contain an 'address' column.")

    if "address" not in followup_df.columns:
        raise ValueError("Follow-up CSV must contain an 'address' column.")

    if "price_usd_initial" not in initial_df.columns:
        raise ValueError("Initial CSV must contain 'price_usd_initial'.")

    if "price_usd_followup" not in followup_df.columns:
        raise ValueError("Follow-up CSV must contain 'price_usd_followup'.")

    # Keep one row per token.
    initial_df = initial_df.drop_duplicates(subset=["address"])
    followup_df = followup_df.drop_duplicates(subset=["address"])

    # Add suffixes so fields from each snapshot stay clear.
    modelling_df = initial_df.merge(
        followup_df,
        on="address",
        how="inner",
        suffixes=("_initial_file", "_followup_file"),
    )

    modelling_df["price_usd_initial"] = pd.to_numeric(
        modelling_df["price_usd_initial"],
        errors="coerce",
    )

    modelling_df["price_usd_followup"] = pd.to_numeric(
        modelling_df["price_usd_followup"],
        errors="coerce",
    )

    # Calculate price change.
    modelling_df["price_change_24h_percent"] = (
        (
            modelling_df["price_usd_followup"]
            - modelling_df["price_usd_initial"]
        )
        / modelling_df["price_usd_initial"]
    ) * 100

    # Create binary target.
    #
    # The project definition is:
    #     pumped = 1 if the token increased by more than 50%
    #     pumped = 0 otherwise
    modelling_df["pumped"] = (
        modelling_df["price_change_24h_percent"] > pump_threshold_percent
    ).astype(int)

    modelling_df["pump_threshold_percent"] = pump_threshold_percent
    modelling_df["label_built_at_utc"] = datetime.now(timezone.utc).isoformat()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

    output_path = (
        PROCESSED_DATA_DIR
        / f"modelling_dataset_pump_threshold_{int(pump_threshold_percent)}_{timestamp}.csv"
    )

    modelling_df.to_csv(output_path, index=False)

    print(f"Saved labelled modelling dataset to: {output_path}")
    print(f"Rows: {len(modelling_df)}")
    print(f"Pumped count: {modelling_df['pumped'].sum()}")
    print(f"Avoid count: {(modelling_df['pumped'] == 0).sum()}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build pump labels from initial and follow-up snapshots."
    )

    parser.add_argument(
        "--initial-csv",
        type=Path,
        required=True,
        help="Path to initial tracking snapshot CSV.",
    )

    parser.add_argument(
        "--followup-csv",
        type=Path,
        required=True,
        help="Path to follow-up snapshot CSV.",
    )

    parser.add_argument(
        "--pump-threshold-percent",
        type=float,
        default=50.0,
        help="Price increase threshold used to define pumped=1.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_pump_labels(
        initial_csv=args.initial_csv,
        followup_csv=args.followup_csv,
        pump_threshold_percent=args.pump_threshold_percent,
    )


if __name__ == "__main__":
    main()