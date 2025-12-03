from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Basic feature helpers ----------

def add_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute price per square foot from price and size.

    Price_per_SqFt (Lakhs per SqFt) = Price_in_Lakhs / Size_in_SqFt
    """
    df = df.copy()

    if "Price_in_Lakhs" not in df.columns or "Size_in_SqFt" not in df.columns:
        raise KeyError("Required columns Price_in_Lakhs or Size_in_SqFt are missing")

    df["Price_per_SqFt"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"].replace(0, np.nan)

    # Replace any inf / NaN back to 0 for safety
    df["Price_per_SqFt"] = df["Price_per_SqFt"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def add_age_of_property(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """
    Ensure Age_of_Property is consistent: recompute from Year_Built.
    """
    df = df.copy()

    if "Year_Built" not in df.columns:
        return df

    df["Age_of_Property"] = current_year - df["Year_Built"]

    return df


# ---------- Encoding categorical / ordinal features ----------

def encode_furnished_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Furnished_Status to numeric:
    Unfurnished -> 0, Semi-furnished -> 1, Furnished -> 2
    """
    df = df.copy()

    mapping = {
        "Unfurnished": 0,
        "Semi-furnished": 1,
        "Furnished": 2,
    }

    if "Furnished_Status" in df.columns:
        df["Furnished_Status_Enc"] = df["Furnished_Status"].map(mapping).fillna(0).astype(int)

    return df


def encode_availability_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Availability_Status to numeric:
    Under_Construction -> 0, Ready_to_Move -> 1
    """
    df = df.copy()

    mapping = {
        "Under_Construction": 0,
        "Ready_to_Move": 1,
    }

    if "Availability_Status" in df.columns:
        df["Availability_Status_Enc"] = df["Availability_Status"].map(mapping).fillna(0).astype(int)

    return df


def encode_transport_and_security(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Public_Transport_Accessibility and Security as ordinal scores:
    Low -> 0, Medium -> 1, High -> 2
    """
    df = df.copy()

    mapping = {"Low": 0, "Medium": 1, "High": 2}

    if "Public_Transport_Accessibility" in df.columns:
        df["Transport_Score"] = df["Public_Transport_Accessibility"].map(mapping).fillna(0).astype(int)

    if "Security" in df.columns:
        df["Security_Score"] = df["Security"].map(mapping).fillna(0).astype(int)

    return df


# ---------- Growth rate & future price ----------

def add_growth_rate_and_future_price(
    df: pd.DataFrame,
    city_col: str = "City",
    price_col: str = "Price_in_Lakhs",
    price_per_sqft_col: str = "Price_per_SqFt",
) -> pd.DataFrame:
    """
    Create a simple city-based annual growth rate and a 5-year future price.

    Idea:
    - Cities with higher median price_per_sqft get slightly higher growth rates.
    - Growth range: 3% to 8% per year.
    """
    df = df.copy()

    if city_col not in df.columns or price_per_sqft_col not in df.columns:
        return df

    # City-wise median price per sqft
    city_median_pps = df.groupby(city_col)[price_per_sqft_col].transform("median")

    # Normalize rank of city median to [0, 1]
    ranks = city_median_pps.rank(method="dense").astype(float)
    norm = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-9)

    # Annual growth between 3% and 8%
    df["Annual_Growth_Rate"] = 0.03 + norm * (0.08 - 0.03)

    if price_col in df.columns:
        df["Future_Price_5Y"] = df[price_col] * (1 + df["Annual_Growth_Rate"]) ** 5

    return df


# ---------- Investment score & label ----------

def add_investment_score_and_label(
    df: pd.DataFrame,
    city_col: str = "City",
    price_col: str = "Price_in_Lakhs",
    price_per_sqft_col: str = "Price_per_SqFt",
    schools_col: str = "Nearby_Schools",
    hospitals_col: str = "Nearby_Hospitals",
    transport_score_col: str = "Transport_Score",
    security_score_col: str = "Security_Score",
    score_col: str = "Investment_Score",
    label_col: str = "Good_Investment",
    min_score_for_good: int = 3,
) -> pd.DataFrame:
    """
    Build a simple rule-based investment score and binary label.

    Rules:
    - Cheaper than city median price_per_sqft -> +1
    - Good public transport (Transport_Score >= 1) -> +1
    - Many schools nearby (Nearby_Schools >= 5) -> +1
    - Many hospitals nearby (Nearby_Hospitals >= 5) -> +1
    - Good security (Security_Score >= 1) -> +1

    Good_Investment = 1 if total score >= min_score_for_good.
    """
    df = df.copy()

    # City-wise median price per sqft
    if city_col in df.columns and price_per_sqft_col in df.columns:
        city_median_pps = df.groupby(city_col)[price_per_sqft_col].transform("median")
    else:
        city_median_pps = pd.Series(0, index=df.index)

    # Conditions
    cond_price = (df[price_per_sqft_col] <= 0.9 * city_median_pps) if price_per_sqft_col in df.columns else False
    cond_transport = (df[transport_score_col] >= 1) if transport_score_col in df.columns else False
    cond_schools = (df[schools_col] >= 5) if schools_col in df.columns else False
    cond_hospitals = (df[hospitals_col] >= 5) if hospitals_col in df.columns else False
    cond_security = (df[security_score_col] >= 1) if security_score_col in df.columns else False

    # Start score at 0
    df[score_col] = 0

    df.loc[cond_price, score_col] += 1
    df.loc[cond_transport, score_col] += 1
    df.loc[cond_schools, score_col] += 1
    df.loc[cond_hospitals, score_col] += 1
    df.loc[cond_security, score_col] += 1

    # Label: 1 = good investment, 0 = otherwise
    df[label_col] = (df[score_col] >= min_score_for_good).astype(int)

    return df


# ---------- Full feature engineering pipeline ----------

def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the dataset.

    This is the main function to call from notebooks.
    """
    df = df.copy()

    df = add_age_of_property(df)
    df = add_price_per_sqft(df)
    df = encode_furnished_status(df)
    df = encode_availability_status(df)
    df = encode_transport_and_security(df)
    df = add_growth_rate_and_future_price(df)
    df = add_investment_score_and_label(df)

    return df
