"""Leakage-safe target construction utilities.

This module centralizes creation of:
- Future_Price_5Y (regression)
- Good_Investment (classification)

Design principles:
- Targets are generated outside feature pipeline (no leakage in src/features)
- Use macro base growth + localized signals (Transport/Security/Investment Score)
- Add controlled Gaussian noise with a seed for reproducibility
- Clip growth to reasonable bounds
- Good_Investment is defined by a relative (quantile) ROI threshold

Assumptions and limitations are documented in the docstrings and must be
recorded in the modeling notebooks for transparency.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


LEAKAGE_PRONE_FEATURES = ["Price_per_SqFt"]


def _safe_divide(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    maxv = series.max()
    if maxv == 0 or np.isnan(maxv):
        return series.fillna(0.0)
    return series / maxv


def generate_targets(
    df: pd.DataFrame,
    years: int = 5,
    base_growth: float = 0.05,
    weights: Optional[Dict[str, float]] = None,
    noise_std: float = 0.03,
    random_state: Optional[int] = 42,
    growth_clip: Tuple[float, float] = (0.03, 0.12),
    investment_quantile: float = 0.65,
) -> pd.DataFrame:
    """Add leakage-aware targets to `df` and return a new DataFrame.

    Future_Price_5Y is computed by applying an *effective growth rate* to the
    current `Price_in_Lakhs`. The effective growth rate is a combination of a
    macro base growth and normalized local signals (transport, security,
    investment_score, recency (age)). Importantly we avoid using `Price_per_SqFt`
    or direct transforms of price as predictive signals for the growth rule to
    reduce the risk of trivial reconstruction.

    The function adds columns:
      - Effective_Growth_Rate
      - Future_Price_5Y
      - ROI
      - Good_Investment (0/1)

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame that must contain at least `Price_in_Lakhs`. Other
        columns used if present: `Transport_Score`, `Security_Score`,
        `Age_of_Property`, `Investment_Score`.
    years: int
        Number of years to project ahead.
    base_growth: float
        Macro baseline growth rate (e.g., 0.05 for 5%).
    weights: dict (optional)
        Weights for local signals. Keys: 'transport', 'security', 'age', 'investment'.
        If omitted, sensible defaults are used.
    noise_std: float
        Standard deviation of Gaussian noise added to growth rate (absolute,
        not relative). Small values (0.02–0.05) are recommended.
    random_state: int or None
        Seed for reproducibility. If None, uses non-deterministic RNG.
    growth_clip: (low, high)
        Min and max allowed effective growth rates.
    investment_quantile: float
        Quantile threshold used to label `Good_Investment` (e.g., 0.65 selects
        the top 35% as good investments).

    Returns
    -------
    pd.DataFrame
        Copy of input with new target columns added.
    """

    if "Price_in_Lakhs" not in df.columns:
        raise ValueError("`Price_in_Lakhs` required to construct Future_Price_5Y")

    df = df.copy()

    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # default weights
    if weights is None:
        weights = {"transport": 0.25, "security": 0.20, "age": 0.25, "investment": 0.30}

    # Normalize available signals safely (0..1)
    transport = _safe_divide(df["Transport_Score"]) if "Transport_Score" in df else pd.Series(0.0, index=df.index)
    security = _safe_divide(df["Security_Score"]) if "Security_Score" in df else pd.Series(0.0, index=df.index)
    age = _safe_divide(1 - (_safe_divide(df["Age_of_Property"]) if "Age_of_Property" in df else pd.Series(0.0, index=df.index)))
    investment = _safe_divide(df["Investment_Score"]) if "Investment_Score" in df else pd.Series(0.0, index=df.index)

    # Weighted signal (does NOT use current price directly)
    signal = (
        weights.get("transport", 0) * transport
        + weights.get("security", 0) * security
        + weights.get("age", 0) * age
        + weights.get("investment", 0) * investment
    )

    # Combine base growth and signal; signal is centered around 0..1, scale it to a sensible delta
    effective_growth = base_growth + signal * (growth_clip[1] - base_growth)

    # Add Gaussian noise to prevent deterministic inversion (absolute noise on growth)
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(df))
    effective_growth = effective_growth + noise

    # Clip to reasonable bounds
    effective_growth = np.clip(effective_growth, growth_clip[0], growth_clip[1])

    df["Effective_Growth_Rate"] = effective_growth

    # Compute future price
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * ((1 + df["Effective_Growth_Rate"]) ** years)

    # Clip unrealistic prices (defensive)
    df["Future_Price_5Y"] = df["Future_Price_5Y"].clip(lower=0)

    # ROI
    df["ROI"] = (df["Future_Price_5Y"] - df["Price_in_Lakhs"]) / df["Price_in_Lakhs"].replace(0, np.nan)

    # Good_Investment by relative threshold (quantile)
    threshold = df["ROI"].quantile(investment_quantile)
    df["Good_Investment"] = (df["ROI"] >= threshold).astype(int)

    return df
