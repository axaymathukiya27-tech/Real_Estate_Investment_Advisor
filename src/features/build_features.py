"""Feature engineering for Real Estate Investment Advisor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime

# ---------- Basic feature helpers ----------

def add_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price per square foot."""
    df = df.copy()

    if "Price_in_Lakhs" not in df.columns or "Size_in_SqFt" not in df.columns:
        raise KeyError("Required columns Price_in_Lakhs or Size_in_SqFt are missing")

    df["Price_per_SqFt"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"].replace(0, np.nan)
    df["Price_per_SqFt"] = (
        df["Price_per_SqFt"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df

def add_age_of_property(df: pd.DataFrame, year_col: str = "Year_Built") -> pd.DataFrame:
    df = df.copy()

    if year_col not in df.columns:
        raise KeyError(f"Column '{year_col}' not found in dataframe")

    current_year = datetime.now().year

    df["Age_of_Property"] = (
        current_year - pd.to_numeric(df[year_col], errors="coerce")
    )

    return df

# ---------- Encoding categorical features ----------

def encode_furnished_status(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Furnished_Status."""
    df = df.copy()

    mapping = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
    if "Furnished_Status" in df.columns:
        df["Furnished_Status_Enc"] = (
            df["Furnished_Status"].map(mapping).fillna(0).astype(int)
        )

    return df


def encode_availability_status(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Availability_Status."""
    df = df.copy()

    mapping = {"Under_Construction": 0, "Ready_to_Move": 1}
    if "Availability_Status" in df.columns:
        df["Availability_Status_Enc"] = (
            df["Availability_Status"].map(mapping).fillna(0).astype(int)
        )

    return df


def encode_transport_and_security(df: pd.DataFrame) -> pd.DataFrame:
    """Encode transport accessibility and security."""
    df = df.copy()

    mapping = {"Low": 0, "Medium": 1, "High": 2}

    if "Public_Transport_Accessibility" in df.columns:
        df["Transport_Score"] = (
            df["Public_Transport_Accessibility"].map(mapping).fillna(0).astype(int)
        )

    if "Security" in df.columns:
        df["Security_Score"] = (
            df["Security"].map(mapping).fillna(0).astype(int)
        )

    return df


# ---------- Feature engineering pipeline ----------
from pathlib import Path
import json

FEATURE_CONFIG_PATH = Path(__file__).parent / "feature_config.json"


def load_feature_config() -> dict:
    """Load the feature configuration JSON.

    This is the single source of truth for the pipeline.
    """
    with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_features(df: pd.DataFrame, require_targets: bool = False) -> None:
    """Validate that the DataFrame contains all features declared in feature_config.json.

    - Checks that all numeric and categorical features listed in the config are
      present as columns in `df`.
    - If ``require_targets`` is True, also verifies targets are present.

    Raises:
        ValueError: with a clear message listing missing features.
    """
    cfg = load_feature_config()

    numeric = set(cfg.get("numeric_features", []))
    categorical = set(cfg.get("categorical_features", []))
    targets = cfg.get("targets", {})

    missing_numeric = [c for c in sorted(numeric) if c not in df.columns]
    missing_categorical = [c for c in sorted(categorical) if c not in df.columns]

    msgs = []
    if missing_numeric:
        msgs.append(f"Missing numeric features: {missing_numeric}")
    if missing_categorical:
        msgs.append(f"Missing categorical features: {missing_categorical}")

    if require_targets:
        missing_targets = [t for t in targets.values() if t and t not in df.columns]
        if missing_targets:
            msgs.append(f"Missing target columns: {missing_targets}")

    # Defensive check: ensure any encoded feature produced by this module is expected in numeric_features
    # Encoded patterns: columns ending with _Enc or containing _Score or specific engineered names
    encoded_candidates = [c for c in df.columns if c.endswith("_Enc") or c.endswith("_Score") or c in {"Price_per_SqFt", "Age_of_Property"}]
    unexpected_encoded = [c for c in encoded_candidates if c not in numeric]
    if unexpected_encoded:
        msgs.append(
            "Unexpected engineered/encoded features present but NOT declared in feature_config.json: "
            f"{unexpected_encoded}. Add them to the config or remove the creation logic in the pipeline."
        )

    if msgs:
        raise ValueError("Feature validation failed:\n" + "\n".join(msgs))


def add_investment_score(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, leakage-free Investment_Score derived from current signals.

    The score is computed from Transport_Score, Security_Score, Age_of_Property,
    Nearby_Schools, Nearby_Hospitals and an amenities count. It is scaled to
    a 1..5 range and does not use any future price information.
    """
    df = df.copy()

    # amenities count (string field like "Playground, Gym")
    if "Amenities" in df.columns:
        df["_amenities_count"] = df["Amenities"].fillna("").apply(
            lambda s: sum(1 for a in s.split(",") if a.strip())
        )
    else:
        df["_amenities_count"] = 0

    # helper to normalize series to [0,1]
    def _norm(series):
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        maxv = s.max() if hasattr(s, "max") else s
        if maxv == 0 or pd.isna(maxv):
            return s * 0.0
        return s / maxv

    def _get_col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    transport_n = _norm(_get_col("Transport_Score"))
    security_n = _norm(_get_col("Security_Score"))
    # Less age is better -> invert normalized age
    age_n = 1 - _norm(_get_col("Age_of_Property"))
    amen_n = _norm(df["_amenities_count"])
    schools_n = _norm(_get_col("Nearby_Schools"))
    hospitals_n = _norm(_get_col("Nearby_Hospitals"))

    # Weighted combination (tunable). Sum of weights = 1.0
    score = (
        0.30 * transport_n
        + 0.25 * security_n
        + 0.20 * age_n
        + 0.15 * amen_n
        + 0.05 * schools_n
        + 0.05 * hospitals_n
    )

    # Scale to 1..5 range
    min_s = score.min()
    max_s = score.max()
    if max_s - min_s <= 1e-9:
        df["Investment_Score"] = 3.0
    else:
        df["Investment_Score"] = ((score - min_s) / (max_s - min_s)) * 4.0 + 1.0

    # drop helpers
    df.drop(columns=["_amenities_count"], inplace=True, errors="ignore")

    return df


def add_annual_growth_rate(df: pd.DataFrame, base_growth: float = 0.05) -> pd.DataFrame:
    """Deterministic proxy for local annual growth computed from Investment_Score.

    This is a deterministic, leakage-safe feature built only from current
    property signals (no future price data). It is intentionally conservative
    (small range) and serves as an optional predictor in downstream models.
    """
    df = df.copy()

    if "Investment_Score" not in df.columns:
        # If investment score is not available, leave column absent (validation
        # will catch missing required features).
        return df

    s = pd.to_numeric(df["Investment_Score"], errors="coerce").fillna(0.0)
    maxv = s.max()
    if maxv == 0 or pd.isna(maxv):
        norm = s * 0.0
    else:
        norm = s / maxv

    # Small multiplier so growth remains around base_growth +/- a little
    df["Annual_Growth_Rate"] = base_growth + norm * 0.02

    return df


def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations only.
    Targets are handled separately to avoid leakage.

    This function applies deterministic, auditable transformations and then
    performs a strict validation pass against `feature_config.json`. Validation
    does NOT require targets here (``require_targets=False``) so this
    function can safely be used in the feature engineering notebook that
    intentionally leaves targets to downstream modeling steps.
    """
    df = df.copy()

    # Derived numeric features
    df = add_age_of_property(df)
    df = add_price_per_sqft(df)

    # Encoded features (only those explicitly expected by feature_config.json)
    df = encode_furnished_status(df)
    df = encode_availability_status(df)
    df = encode_transport_and_security(df)

    # Deterministic, leakage-free derived features required by config
    df = add_investment_score(df)
    df = add_annual_growth_rate(df)

    # Final fast validation: verify that all features declared in the config exist
    # NOTE: We do not require targets here (require_targets=False) because targets
    # are created in modeling notebooks to avoid leakage.
    validate_features(df, require_targets=False)

    return df
