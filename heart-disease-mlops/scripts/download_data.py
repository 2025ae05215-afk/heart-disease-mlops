"""Download the Heart Disease UCI (Cleveland) dataset -> data/heart_raw.csv.

Tries the official UCI file first; falls back to a public CSV mirror.
Output schema: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,
slope,ca,thal,target  (target: 1 = heart disease present, 0 = absent).
"""
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_PATH  # noqa: E402

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
MIRROR_URL = ("https://raw.githubusercontent.com/sharmaroshan/"
              "Heart-UCI-Dataset/master/heart.csv")
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def from_uci() -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(_get(UCI_URL)), header=None, names=COLS,
                     na_values="?")
    df["target"] = (df["target"] > 0).astype(int)  # 0..4 -> binary
    return df


def from_local(path: str) -> pd.DataFrame:
    """Parse a local copy of processed.cleveland.data (UCI original)."""
    df = pd.read_csv(path, header=None, names=COLS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    return df


def from_mirror() -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(_get(MIRROR_URL)), encoding="utf-8-sig")
    # mirror encodes target inversely to the UCI convention
    df["target"] = 1 - df["target"]
    return df[COLS]


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:  # python scripts/download_data.py path/to/processed.cleveland.data
        df, src = from_local(sys.argv[1]), "local UCI file"
    else:
        df, src = _remote()
    df.to_csv(RAW_PATH, index=False)
    print(f"[{src}] saved {df.shape} -> {RAW_PATH}")


def _remote():
    try:
        return from_uci(), "UCI"
    except Exception as exc:  # network / blocked
        print(f"UCI download failed ({exc!r}); using mirror")
        return from_mirror(), "mirror"


if __name__ == "__main__":
    main()
