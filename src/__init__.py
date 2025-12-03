"""
Root package for Real Estate ML Project.

This allows importing modules like:
    from src.data import load_raw_data
    from src.features import run_feature_pipeline
    from src.models import make_predictions
"""

# Expose subpackages for easier access
from .data import *
from .features import *
from .models import *
