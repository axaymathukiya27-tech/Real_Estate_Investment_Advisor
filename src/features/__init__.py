# src/features/__init__.py
"""
Package initializer for src.features

This file intentionally left minimal so that imports like:
    from src.features.build_features import run_feature_pipeline
or
    from .build_features import run_feature_pipeline
work reliably when the package is imported.
"""
from .build_features import run_feature_pipeline

__all__ = ["run_feature_pipeline"]
