"""MLflow utilities for standardized experiment setup and reproducibility.

This module provides:
- setup_mlflow: initialize MLflow tracking URI and experiment (backward compatible)
- set_mlflow_experiment: explicit helper to set tracking URI and experiment
- log_requirements: log `requirements.txt` as an MLflow artifact if present

It aims to ensure all runs share a consistent experiment and that environment
metadata is saved for reproducibility.
"""
from __future__ import annotations

import os
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLRUNS = PROJECT_ROOT / "mlruns"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def setup_mlflow(tracking_uri: str | None = None, experiment_name: str | None = None):
    """Initialize MLflow (backwards compatible with previous implementation).

    If env vars are set (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT) they take
    precedence. Returns (tracking_uri, experiment_name).
    """
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", None)
    experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT", "real_estate_investment")

    if tracking_uri is None:
        # default to file-backed mlruns directory
        tracking_uri = f"file:///{DEFAULT_MLRUNS.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    # ensure experiment exists
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        pass

    mlflow.set_experiment(experiment_name)
    return tracking_uri, experiment_name


def set_mlflow_experiment(experiment_name: str = "real_estate_investment", mlruns_path: Path | None = None):
    """Convenience wrapper to set file-backed mlruns and the experiment name."""
    if mlruns_path is None:
        mlruns_path = DEFAULT_MLRUNS
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"file:///{mlruns_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri, experiment_name


def log_requirements():
    """Log requirements.txt to the active MLflow run (best-effort).

    Call this from inside an active run. If the file is missing, set a tag so
    the absence is recorded.
    """
    if REQUIREMENTS_FILE.exists():
        try:
            mlflow.log_artifact(str(REQUIREMENTS_FILE))
        except Exception:
            # best-effort: don't raise
            mlflow.set_tag("requirements_logged", "false")
    else:
        mlflow.set_tag("requirements_logged", "false")

