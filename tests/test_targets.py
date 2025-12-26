import pandas as pd
import pytest
from src.targets import generate_targets


def test_generate_targets_basic():
    df = pd.DataFrame({"Price_in_Lakhs": [100.0, 200.0, 300.0],
                       "Transport_Score": [2, 1, 0],
                       "Security_Score": [1, 2, 0],
                       "Age_of_Property": [10, 20, 5],
                       "Nearby_Schools": [2, 0, 5],
                       "Nearby_Hospitals": [1, 2, 1],
                       "Amenities": ["Pool", "Gym, Pool", ""]})
    out = generate_targets(df.copy(), random_state=0)
    assert "Future_Price_5Y" in out.columns
    assert "Good_Investment" in out.columns
    assert (out["Future_Price_5Y"] > out["Price_in_Lakhs"]).all()


def test_generate_targets_requires_price():
    df = pd.DataFrame({"Transport_Score": [1]})
    with pytest.raises(ValueError):
        generate_targets(df)


def test_generate_targets_reproducible():
    df = pd.DataFrame({"Price_in_Lakhs": [100.0, 150.0]})
    a = generate_targets(df.copy(), random_state=42)
    b = generate_targets(df.copy(), random_state=42)
    # with same seed, outputs should match
    assert a["Future_Price_5Y"].tolist() == b["Future_Price_5Y"].tolist()
