"""
wds.py

Whale Dominance Score calculation.

WDS is calculated from top-holder ownership shares.

For each holder:
    ownership_share = holder_balance / supply

Then:
    C = sum of top-holder shares
    H = sum of squared top-holder shares
    N = normalized concentration score
    WDS = C * N
"""

from birdeye_client import safe_float


def extract_holder_balances(holder_rows: list[dict]) -> list[float]:
    """
    Extract ui_amount balances from Birdeye holder rows.

    Birdeye's holder endpoint test showed that ui_amount is already adjusted
    for token decimals, so it is the clean balance field to use.
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

    Parameters
    ----------
    balances:
        Holder balances from the holder endpoint.

    supply:
        Circulating supply is preferred.
        Total supply can be used as a fallback.

    Returns
    -------
    dict:
        WDS result and formula components.
    """

    supply = safe_float(supply)

    if supply is None or supply <= 0:
        return {
            "can_calculate_wds": False,
            "wds_reason": "Missing or invalid supply.",
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
            "wds_reason": "Need at least 2 holder balances.",
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
            "wds_reason": "Cumulative holder share is zero.",
            "wds_n_used": n,
            "wds_c": C,
            "wds_h": None,
            "wds_n": None,
            "wds": None,
        }

    H = sum(share ** 2 for share in shares)

    # Normalized concentration score.
    N = ((H / (C ** 2)) - (1 / n)) / (1 - (1 / n))

    WDS = C * N

    return {
        "can_calculate_wds": True,
        "wds_reason": None,
        "wds_n_used": n,
        "wds_c": C,
        "wds_h": H,
        "wds_n": N,
        "wds": WDS,
    }