import pandas as pd
import pytest
from src.features.build_features import validate_features, load_feature_config


def make_minimal_df():
    cfg = load_feature_config()
    # create a small df with required numeric and categorical columns
    numeric = cfg["numeric_features"]
    categorical = cfg["categorical_features"]

    data = {c: 0 for c in numeric}
    data.update({c: "X" for c in categorical})
    df = pd.DataFrame([data])
    return df


def test_validate_features_pass():
    df = make_minimal_df()
    # should not raise
    validate_features(df, require_targets=False)


def test_validate_features_missing_numeric():
    df = make_minimal_df()
    df = df.drop(columns=[df.columns[0]])
    with pytest.raises(ValueError):
        validate_features(df, require_targets=False)


def test_validate_features_unexpected_engineered():
    df = make_minimal_df()
    # Add an engineered column that is not declared
    df["New_Eng_Score"] = 1
    with pytest.raises(ValueError):
        validate_features(df, require_targets=False)
