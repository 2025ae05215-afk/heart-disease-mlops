import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.data import clean, split_xy
from src.features import build_preprocessor


def make_df(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(30, 70, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n), "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(90, 190, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(0, 3, n), "ca": rng.integers(0, 4, n),
        "thal": rng.integers(0, 4, n), "target": rng.integers(0, 5, n)})


def test_clean_imputes_and_binarizes():
    df = make_df()
    df.loc[0, "chol"] = np.nan
    df.loc[1, "ca"] = np.nan
    out = clean(df)
    assert out.isna().sum().sum() == 0
    assert set(out[TARGET].unique()) <= {0, 1}


def test_clean_missing_column_raises():
    with pytest.raises(ValueError):
        clean(make_df().drop(columns=["age"]))


def test_clean_drops_duplicates():
    df = make_df()
    assert len(clean(pd.concat([df, df.iloc[:5]]))) <= len(df)


def test_preprocessor_output_numeric_and_handles_unseen():
    X, _ = split_xy(clean(make_df()))
    prep = build_preprocessor().fit(X)
    Xt = prep.transform(X)
    assert Xt.shape[0] == len(X) and np.isfinite(Xt).all()
    unseen = X.iloc[:1].copy()
    unseen["cp"] = 99
    assert prep.transform(unseen).shape[1] == Xt.shape[1]


def test_features_list():
    assert len(FEATURES) == 13
