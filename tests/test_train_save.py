import sys
import types
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import run_feature_pipeline
from src.models.train import train_baseline_models
from src.config import BEST_CLASSIFIER, BEST_REGRESSOR


def make_dataset(n=120):
    df = pd.DataFrame({
        "BHK": [2] * n,
        "Size_in_SqFt": [1000] * n,
        "Price_in_Lakhs": [50.0] * n,
        "Year_Built": [2010] * n,
        "Floor_No": [2] * n,
        "Total_Floors": [5] * n,
        "Nearby_Schools": [1] * n,
        "Nearby_Hospitals": [1] * n,
        "Furnished_Status": ["Unfurnished"] * n,
        "Public_Transport_Accessibility": ["Medium"] * n,
        "Parking_Space": ["No"] * n,
        "Security": ["Low"] * n,
        "Amenities": [""] * n,
        "Facing": ["East"] * n,
        "Owner_Type": ["Owner"] * n,
        "Availability_Status": ["Ready_to_Move"] * n,
        # required categorical fields
        "State": ["StateX"] * n,
        "City": ["CityX"] * n,
        "Locality": ["LocalityX"] * n,
        "Property_Type": ["Apartment"] * n,
    })

    # add targets
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * 1.1
    df["Good_Investment"] = (np.arange(n) % 3 == 0).astype(int)

    # run feature pipeline to ensure derived cols exist
    df = run_feature_pipeline(df)
    return df


def test_train_baseline_saves_models(tmp_path, monkeypatch):
    # Inject a dummy mlflow with a start_run context manager to avoid dependency on mlflow server
    def start_run(run_name=None):
        @contextlib.contextmanager
        def _cm():
            yield None
        return _cm()

    dummy_mlflow = types.SimpleNamespace(start_run=start_run, log_metrics=lambda d: None, log_param=lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "mlflow", dummy_mlflow)

    # Inject a dummy src.mlflow_setup module if not importable in test env
    dummy_mlflow_setup = types.SimpleNamespace(set_mlflow_experiment=lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "src.mlflow_setup", dummy_mlflow_setup)

    model_dir = tmp_path / "models"

    df = make_dataset(120)

    clf_models, reg_models, clf_metrics, reg_metrics = train_baseline_models(df, model_dir=str(model_dir))

    clf_path = model_dir / BEST_CLASSIFIER
    reg_path = model_dir / BEST_REGRESSOR

    assert clf_path.exists(), f"Expected classifier to be saved at {clf_path}"
    assert reg_path.exists(), f"Expected regressor to be saved at {reg_path}"

    assert isinstance(clf_metrics, dict) and isinstance(reg_metrics, dict)
    assert len(clf_metrics) > 0 and len(reg_metrics) > 0
