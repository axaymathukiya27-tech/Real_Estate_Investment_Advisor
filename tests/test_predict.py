import pandas as pd
import pytest
from src.models.predict import load_model, load_trained_models, make_predictions
from pathlib import Path


def test_load_model_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_model("models/non_existent_model.pkl")


def test_make_predictions_input_validation():
    # input without required raw columns should raise during feature pipeline/validation
    clf = None
    reg = None
    with pytest.raises(Exception):
        make_predictions(clf, reg, pd.DataFrame({}))


def test_load_trained_models_missing_paths():
    with pytest.raises(FileNotFoundError):
        load_trained_models("models/does_not_exist_clf.pkl", "models/does_not_exist_reg.pkl")
