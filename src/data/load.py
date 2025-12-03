
from pathlib import Path
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw housing dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df
