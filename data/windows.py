# /data/windows.py
import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

NON_NUMERIC_DROP_CANDIDATES = {"date", "Date", "timestamp", "Timestamp", "Unnamed: 0", "index", "Index"}

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """If a date-like column exists, set it as a DateTimeIndex; otherwise keep existing index."""
    for col in list(df.columns):
        if col in NON_NUMERIC_DROP_CANDIDATES:
            try:
                idx = pd.to_datetime(df[col], errors="coerce")
                if idx.notna().all():
                    df = df.drop(columns=[col])
                    df.index = idx
                    df = df.sort_index()
                    return df
            except Exception:
                pass
    # already indexed or no date column; just return
    return df

def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all non-numeric columns (keeps only float/int)."""
    num = df.select_dtypes(include=["number"]).copy()
    return num

def _read_asset(path: str, features: Optional[List[str]]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # Read
    Xtr = pd.read_csv(os.path.join(path, "X_train.csv"))
    ytr = pd.read_csv(os.path.join(path, "y_train.csv"))
    Xte = pd.read_csv(os.path.join(path, "X_test.csv"))
    yte = pd.read_csv(os.path.join(path, "y_test.csv"))

    # Set datetime index if a date/timestamp column exists, then drop it
    Xtr = _ensure_datetime_index(Xtr)
    Xte = _ensure_datetime_index(Xte)
    ytr = _ensure_datetime_index(ytr)
    yte = _ensure_datetime_index(yte)

    # Keep only numeric columns; y may be a single numeric column
    Xtr_num = _numeric_only(Xtr)
    Xte_num = _numeric_only(Xte)

    # Feature subset (only among numeric columns)
    if features is not None:
        keep = [c for c in features if c in Xtr_num.columns]
        if not keep:
            raise ValueError(f"None of requested features {features} found in {path}/X_*.csv numeric columns {list(Xtr_num.columns)}")
        Xtr_num = Xtr_num[keep]
        Xte_num = Xte_num[keep]

    # y: take first numeric column
    ytr_num = _numeric_only(ytr)
    yte_num = _numeric_only(yte)
    if ytr_num.shape[1] == 0 or yte_num.shape[1] == 0:
        raise ValueError(f"No numeric target column found in {path}/y_*.csv")
    ytr_series = ytr_num.iloc[:, 0]
    yte_series = yte_num.iloc[:, 0]

    return Xtr_num, ytr_series, Xte_num, yte_series

def _align_by_index(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join all frames by their indices (dates)."""
    # Start with the first
    it = iter(frames.values())
    aligned = next(it)
    for df in it:
        aligned = aligned.join(df, how="inner")
    return aligned

def _build_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xw, yw = [], []
    for i in range(len(X) - seq_len):
        Xw.append(X[i:i+seq_len])
        yw.append(y[i+seq_len])
    return np.asarray(Xw, dtype=np.float32), np.asarray(yw, dtype=np.float32)

def load_windows(
    seq_len: int = 60,
    features: Optional[List[str]] = None,
    processed_dir: str = "processed",
    val_ratio_in_train: float = 0.10,   # paper: “last portion” of train
):
    """
    Builds:
      - Train windows from TRAIN csvs (2010–2017),
      - Validation = LAST val_ratio_in_train of train windows,
      - Test windows from TEST csvs (2018–2020).
    Output shapes:
      X_* : (N, seq_len, n_assets * F)
      y_* : (N, n_assets)
    """
    # Discover assets
    assets = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])

    Xtr_per_asset, ytr_per_asset = {}, {}
    Xte_per_asset, yte_per_asset = {}, {}

    for asset in assets:
        path = os.path.join(processed_dir, asset)
        Xtr_df, ytr_ser, Xte_df, yte_ser = _read_asset(path, features)

        # Prefix columns with asset for uniqueness before alignment
        Xtr_per_asset[asset] = Xtr_df.add_prefix(f"{asset}__")
        Xte_per_asset[asset] = Xte_df.add_prefix(f"{asset}__")
        ytr_per_asset[asset] = ytr_ser.rename(f"{asset}")
        yte_per_asset[asset] = yte_ser.rename(f"{asset}")

    # Align by index (dates) across all assets (inner join)
    Xtr_aligned = _align_by_index(Xtr_per_asset)   # columns: asset__feature
    Xte_aligned = _align_by_index(Xte_per_asset)
    ytr_aligned = _align_by_index({k: v.to_frame() for k, v in ytr_per_asset.items()})
    yte_aligned = _align_by_index({k: v.to_frame() for k, v in yte_per_asset.items()})

    # Convert to numpy
    # X_* now has columns grouped by asset and feature; we keep them flattened (n_assets * F)
    Xtr_np = Xtr_aligned.to_numpy(dtype=np.float32)
    Xte_np = Xte_aligned.to_numpy(dtype=np.float32)
    ytr_np = ytr_aligned.to_numpy(dtype=np.float32)   # (T_tr, n_assets)
    yte_np = yte_aligned.to_numpy(dtype=np.float32)   # (T_te, n_assets)

    # Build rolling windows
    Xtr_win, ytr_win = _build_windows(Xtr_np, ytr_np, seq_len)
    Xte_win, yte_win = _build_windows(Xte_np, yte_np, seq_len)

    # Validation = last slice of train windows
    n_val = max(1, int(val_ratio_in_train * len(Xtr_win)))
    X_val, y_val = Xtr_win[-n_val:], ytr_win[-n_val:]
    X_train, y_train = Xtr_win[:-n_val], ytr_win[:-n_val]

    # Reshape X_* to (N, seq_len, n_assets*F)
    # (Already this shape since we flattened features when making numpy arrays.)
    return X_train, y_train, X_val, y_val, Xte_win, yte_win
