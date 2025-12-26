from pathlib import Path
import sys
PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.features.build_features import validate_features

print('Loading processed dataset...')
df = pd.read_csv('data/processed/housing_with_features.csv', nrows=50)
print('Columns present:', list(df.columns))

validate_features(df, require_targets=False)
print('Validation OK: all declared features exist and no unexpected engineered features')
