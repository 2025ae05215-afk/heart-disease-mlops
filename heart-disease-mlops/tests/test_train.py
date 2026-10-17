from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.data import clean, split_xy
from src.features import build_preprocessor
from tests.test_data import make_df


def test_pipeline_trains_and_beats_chance():
    X, _ = split_xy(clean(make_df(120)))
    y = (X["thalach"] < 140).astype(int)  # learnable signal
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))])
    assert cross_val_score(pipe, X, y, cv=3, scoring="roc_auc").mean() > 0.6
