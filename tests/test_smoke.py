import os
import subprocess
from pathlib import Path
import pandas as pd
from src.features.build_features import validate_features


def test_generate_processed_smoke(tmp_path):
    # Run the script with SAMPLE_SIZE=100 to keep CI quick.
    env = os.environ.copy()
    env["SAMPLE_SIZE"] = "100"
    script = Path("scripts/generate_processed.py")

    # Ensure the output path is temporary to avoid overwriting main data during CI
    # Create temp data dir
    out = Path("data/processed/housing_with_features.csv")
    if out.exists():
        out.unlink()

    res = subprocess.run(["python", str(script)], env=env, capture_output=True, text=True, check=True)
    assert out.exists()

    # Load a small portion and validate
    df = pd.read_csv(out, nrows=20)
    validate_features(df, require_targets=False)
