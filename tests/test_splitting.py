import pandas as pd
from src.models.train import split_data_for_modeling
from src.features.build_features import run_feature_pipeline


def make_dataset(n=200):
    # create a minimal synthetic dataset with features required
    df = pd.DataFrame({
        "BHK": [2]*n,
        "Size_in_SqFt": [1000]*n,
        "Price_in_Lakhs": [50.0]*n,
        "Year_Built": [2010]*n,
        "Floor_No": [2]*n,
        "Total_Floors": [5]*n,
        "Nearby_Schools": [1]*n,
        "Nearby_Hospitals": [1]*n,
        "Furnished_Status": ["Unfurnished"]*n,
        "Public_Transport_Accessibility": ["Medium"]*n,
        "Parking_Space": ["No"]*n,
        "Security": ["Low"]*n,
        "Amenities": [""]*n,
        "Facing": ["East"]*n,
        "Owner_Type": ["Owner"]*n,
        "Availability_Status": ["Ready_to_Move"]*n,
        # required categorical fields
        "State": ["StateX"]*n,
        "City": ["CityX"]*n,
        "Locality": ["LocalityX"]*n,
        "Property_Type": ["Apartment"]*n,
    })

    # add targets
    import numpy as np
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * 1.1
    # Create a classification target with some imbalance
    df["Good_Investment"] = (np.arange(n) % 3 == 0).astype(int)

    # run feature pipeline to ensure derived cols exist
    df = run_feature_pipeline(df)
    return df


def test_split_returns_consistent_shapes():
    df = make_dataset(120)
    (X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test, numeric_cols, categorical_cols) = split_data_for_modeling(df)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_clf_test) == len(y_reg_test) == len(X_test)


def test_split_fallback_when_single_class():
    df = make_dataset(50)
    df["Good_Investment"] = 0  # single class -> stratify will fail and fallback to random
    # should not raise
    split_data_for_modeling(df)
