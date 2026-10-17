"""Data loading and cleaning."""
import pandas as pd

from .config import CATEGORICAL, CLEAN_PATH, FEATURES, NUMERIC, RAW_PATH, TARGET


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce types, impute missing values, validate target.

    Numeric -> median; categorical -> mode. Target forced to {0, 1}.
    """
    df = df.copy().drop_duplicates().reset_index(drop=True)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for col in NUMERIC + CATEGORICAL:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in NUMERIC:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL:
        df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(int)
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df[FEATURES + [TARGET]]


def save_clean(path=CLEAN_PATH) -> pd.DataFrame:
    df = clean(load_raw())
    df.to_csv(path, index=False)
    return df


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]
