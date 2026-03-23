import pandas as pd


def load_data(path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(path, encoding=encoding)
