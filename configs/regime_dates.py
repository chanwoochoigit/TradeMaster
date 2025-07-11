"""
MCAD Regime Date Configurations
==============================

Gold standard date ranges for all MCAD trading regimes.
This is the single source of truth for all date configurations.

Usage:
    from configs.regime_dates import REGIME_DATES

    dates = REGIME_DATES["covid"]
    train_start = dates["train_start"]
"""

REGIME_DATES = {
    "trade_war_i": {
        "train_start": "2006-09-06",
        "train_end": "2015-07-01",
        "val_start": "2015-07-01",
        "val_end": "2018-01-01",
        "test_start": "2018-01-01",
        "test_end": "2020-01-01",
    },
    "covid": {
        "train_start": "2006-09-06",
        "train_end": "2017-07-01",
        "val_start": "2017-07-01",
        "val_end": "2020-01-01",
        "test_start": "2020-01-01",
        "test_end": "2022-09-01",
    },
    "trade_war": {
        "train_start": "2006-09-06",
        "train_end": "2021-01-01",
        "val_start": "2021-01-01",
        "val_end": "2023-06-01",
        "test_start": "2023-06-01",
        "test_end": "2025-06-01",
    },
}


def get_regime_dates(regime_name: str) -> dict:
    """
    Get date configuration for a specific regime.

    Args:
        regime_name: One of 'covid', 'trade_war', 'trade_war_i'

    Returns:
        Dict containing train_start, train_end, val_start, val_end,
        test_start, test_end dates

    Raises:
        KeyError: If regime_name is not recognized
    """
    if regime_name not in REGIME_DATES:
        available = ", ".join(REGIME_DATES.keys())
        raise KeyError(f"Unknown regime '{regime_name}'. Available: {available}")

    return REGIME_DATES[regime_name].copy()


def get_all_regimes() -> list:
    """Get list of all available regime names."""
    return list(REGIME_DATES.keys())


def validate_regime_dates() -> bool:
    """
    Validate that all regime date ranges are consistent and logical.

    Returns:
        True if all dates are valid, False otherwise
    """
    from datetime import datetime

    for regime, dates in REGIME_DATES.items():
        try:
            # Parse all dates
            train_start = datetime.strptime(dates["train_start"], "%Y-%m-%d")
            train_end = datetime.strptime(dates["train_end"], "%Y-%m-%d")
            val_start = datetime.strptime(dates["val_start"], "%Y-%m-%d")
            val_end = datetime.strptime(dates["val_end"], "%Y-%m-%d")
            test_start = datetime.strptime(dates["test_start"], "%Y-%m-%d")
            test_end = datetime.strptime(dates["test_end"], "%Y-%m-%d")

            # Validate chronological order
            if not (
                train_start < train_end <= val_start < val_end <= test_start < test_end
            ):
                print(f"❌ Invalid date sequence for {regime}")
                return False

            print(f"✅ {regime}: {dates['train_start']} → {dates['test_end']}")

        except ValueError as e:
            print(f"❌ Invalid date format in {regime}: {e}")
            return False

    return True


if __name__ == "__main__":
    print("MCAD Regime Date Validation")
    print("=" * 40)

    if validate_regime_dates():
        print("\n🎉 All regime dates are valid!")
    else:
        print("\n💥 Date validation failed!")
        exit(1)
