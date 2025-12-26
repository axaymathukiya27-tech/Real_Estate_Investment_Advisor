"""Generate processed dataset by applying run_feature_pipeline to raw CSV.

Saves output to data/processed/housing_with_features.csv (overwrites existing file).
"""
from pathlib import Path
import sys
from pathlib import Path as _P
# ensure project root is on sys.path so `src` package is importable when running as script
PROJECT_ROOT = str(_P(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.load import load_raw_data
from src.features.build_features import run_feature_pipeline

RAW = Path("data/raw/india_housing_prices.csv")
OUT = Path("data/processed/housing_with_features.csv")

import os

print("Loading raw data from:", RAW)
raw = load_raw_data(str(RAW))
print("Raw rows:", len(raw))

# Allow CI/dev to limit sample size via SAMPLE_SIZE env var for faster runs
sample_size = os.getenv("SAMPLE_SIZE")
if sample_size is not None:
    try:
        n = int(sample_size)
        if n < len(raw):
            print(f"Sampling {n} rows from raw for a faster run (SAMPLE_SIZE={n})")
            raw = raw.sample(n, random_state=42).reset_index(drop=True)
    except Exception:
        print("Warning: SAMPLE_SIZE set but invalid; ignoring and using full dataset")

print("Applying feature pipeline...")
df_proc = run_feature_pipeline(raw)
print("Processed rows:", len(df_proc))

OUT.parent.mkdir(parents=True, exist_ok=True)
df_proc.to_csv(OUT, index=False)
print("Saved processed dataset to:", OUT)
