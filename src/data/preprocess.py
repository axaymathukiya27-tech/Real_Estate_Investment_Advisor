from __future__ import annotations

from typing import Sequence
import pandas as pd


# Columns where we will cap extreme values
OUTLIER_COLS = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]


def cap_outliers(
    df: pd.DataFrame,
    columns: Sequence[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Cap extreme values in the given numeric columns using quantiles.

    Example: lower_q=0.01, upper_q=0.99 → clip below 1st and above 99th percentile.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            # silently skip if column not present
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            # don't try to cap non-numeric columns
            continue

        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        df[col] = df[col].clip(lower, upper)

    return df


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for the raw housing data.

    Steps:
    - Drop duplicate rows (safety).
    - Cap outliers in price, size and price_per_sqft.
    - (We don't handle missing values here because EDA showed 0 missing.)
    """
    df = df.copy()

    # Remove duplicates if any appear later
    df = df.drop_duplicates()

    # Cap extreme values in selected numeric columns
    df = cap_outliers(df, OUTLIER_COLS, lower_q=0.01, upper_q=0.99)

    return df
