# Reproducibility & MLflow Logging

This document explains how MLflow is configured for this project and what is
logged to ensure reproducibility of experiments and model artifacts.

## MLflow: standardization

- **Experiment name**: `real_estate_investment` (used for all baseline and tuning runs)
- Use the helper in `src/models/mlflow_setup.py` to initialize or customize MLflow:

```python
from src.models.mlflow_setup import set_mlflow_experiment, log_requirements
set_mlflow_experiment("real_estate_investment")
```

By default the project uses a file-backed `mlruns/` directory at the repository
root, unless the environment variable `MLFLOW_TRACKING_URI` is set.

## What is logged for each run

Every run logs the following metadata (where applicable):

- `random_state` (seed used for splits and stochastic procedures)
- `test_size` (holdout fraction)
- `cv_object` (type used for CV; e.g. `StratifiedKFold` or `KFold`)
- `feature_count` (number of features used by the model)
- `n_rows`, `n_rows_train`, `n_rows_test`
- `run_type` tag (`baseline` or `tuning`)

Additionally, MLflow runs include:

- Best hyperparameters (for tuning runs)
- CV mean & std for the best candidate (`best_cv_mean`, `best_cv_std`)
- Final test metrics (classification: `f1`, `precision`, `recall`, `roc_auc`; regression: `mae`, `rmse`, `r2`)
- Tuned or baseline model artifacts (saved with canonical names such as `tuned_classification_model.pkl`)
- The project's `requirements.txt` (logged as an artifact when present)

## How to reproduce a run locally

1. Create a Python environment per `requirements.txt`.
2. Run your notebook (e.g., `03_model_baseline.ipynb`) or call the training API:
   - `python -m src.models.train` (if exposed) or run functions from `src.models`.
3. Open the MLflow UI:

```bash
mlflow ui --backend-store-uri file:///<path-to-repo>/mlruns
```

4. Inspect experiments under `real_estate_investment` and compare runs using
   `run_type` and the logged `best_cv_mean` and `best_cv_std` metrics.

## Notes

- The project logs `requirements.txt` to each run (best-effort). If missing, a
  tag `requirements_logged=false` is set on the run.
- For deterministic behavior, set `RANDOM_STATE` in `src/config.py` or pass a
  `random_state` argument to the tuning functions.
