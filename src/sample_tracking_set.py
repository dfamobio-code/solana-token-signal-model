"""
sample_tracking_set.py

Randomly select tokens from a candidate pool.

Purpose:
    Select 200 tokens from the 1000-token candidate pool using a fixed seed.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def sample_tracking_set(input_csv: Path, sample_size: int, seed: int) -> Path:
    """
    Randomly sample a tracking set from the candidate pool.
    """

    df = pd.read_csv(input_csv)

    if "address" not in df.columns:
        raise ValueError("Input CSV must contain an 'address' column.")

    df = df.dropna(subset=["address"])
    df = df.drop_duplicates(subset=["address"])

    if len(df) < sample_size:
        raise ValueError(
            f"Cannot sample {sample_size} tokens from only {len(df)} candidates."
        )

    tracking_df = df.sample(
        n=sample_size,
        random_state=seed,
    ).copy()

    sampled_at_utc = datetime.now(timezone.utc).isoformat()

    tracking_df["selected_for_tracking"] = True
    tracking_df["sample_size"] = sample_size
    tracking_df["random_seed"] = seed
    tracking_df["sampled_at_utc"] = sampled_at_utc

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

    output_path = RAW_DATA_DIR / f"tracking_set_{timestamp}_seed{seed}.csv"

    tracking_df.to_csv(output_path, index=False)

    print(f"Saved tracking set to: {output_path}")
    print(f"Tracking token count: {len(tracking_df)}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample selected tracking tokens from a candidate pool."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to candidate pool CSV.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of tokens to sample.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260601,
        help="Random seed for reproducible sampling.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    sample_tracking_set(
        input_csv=args.input_csv,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()