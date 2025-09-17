# process_pipeline.py
import os
import json
import pandas as pd
import numpy as np
import shutil

# ---------- Feature helpers ----------

def reset(base_dir="processed"):
    """
    Reset the datasets folder:
    1. Delete the datasets folder if it exists
    2. Create datasets/
    3. Create datasets/training_data
    4. Create datasets/testing_data
    """
    # 1. Delete datasets folder if exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"🗑️ Removed existing {base_dir}/")

    # 2. Create datasets folder
    os.makedirs(base_dir, exist_ok=True)
    print(f"📂 Created {base_dir}/")


def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns: ln(Pt / Pt-1)."""
    return np.log(close / close.shift(1))

def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling standard deviation of returns."""
    return returns.rolling(window).std()

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing.
    Works for any price-like series (including yields).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's RMA via EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Edge cases: if avg_loss == 0 -> RSI = 100; if avg_gain == 0 -> RSI = 0
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi = rsi.where(avg_gain > 0, 0.0)

    return rsi

def build_features_from_close(close: pd.Series) -> pd.DataFrame:
    """
    Given a 'Close' series indexed by date, compute:
      - ret_1d: daily log return
      - ma5_ret, ma21_ret: moving average of returns (5, 21)
      - vol21: 21-day rolling std of returns
      - rsi14: RSI on price (14)
    Returns a DataFrame aligned to the original index (with NaNs at the start from rolling windows).
    """
    features = pd.DataFrame(index=close.index)
    features["ret_1d"]   = log_returns(close)
    features["ma5_ret"]  = features["ret_1d"].rolling(5).mean()
    features["ma21_ret"] = features["ret_1d"].rolling(21).mean()
    features["vol21"]    = rolling_volatility(features["ret_1d"], window=21)
    features["rsi14"]    = rsi_wilder(close, period=14)
    return features

def standardize_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit z-score scaling on training set only, then apply to both train and test.
    Returns (X_train_scaled, X_test_scaled, scaler_dict) where scaler_dict holds means/stds.
    """
    means = X_train.mean()
    stds  = X_train.std(ddof=0).replace(0, 1.0)  # avoid divide-by-zero

    X_train_scaled = (X_train - means) / stds
    X_test_scaled  = (X_test  - means) / stds

    scaler = {
        "means": means.to_dict(),
        "stds":  stds.to_dict(),
        "columns": list(X_train.columns),
    }
    return X_train_scaled, X_test_scaled, scaler

# ---------- Main per-asset processor ----------

def process_asset(name: str,
                  in_base: str = "datasets",
                  out_base: str = "processed",
                  drop_na_after_features: bool = True) -> None:
    """
    Build model-ready features for one asset.
    Expects:
      datasets/training_data/{name}.csv  (with 'Close' column)
      datasets/testing_data/{name}.csv   (with 'Close' column)
    Produces:
      processed/{name}/X_train.csv, y_train.csv
      processed/{name}/X_test.csv,  y_test.csv
      processed/{name}/scaler.json
    Target: next-day log return (y_next_ret).
    """
    train_path = os.path.join(in_base, "training_data", f"{name}.csv")
    test_path  = os.path.join(in_base, "testing_data",  f"{name}.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Missing CSVs for {name}. Expected:\n  {train_path}\n  {test_path}")
        return

    # Load
    train = pd.read_csv(train_path, parse_dates=True, index_col=0)
    test  = pd.read_csv(test_path,  parse_dates=True, index_col=0)

    # Basic sanity
    for df, label in [(train, "train"), (test, "test")]:
        if "Close" not in df.columns:
            raise ValueError(f"{name} {label} CSV must contain a 'Close' column.")

    # Build features separately on train/test (no leakage in temporal ops)
    feats_train = build_features_from_close(train["Close"])
    feats_test  = build_features_from_close(test["Close"])

    # Target = next-day return (computed from the same Close used for features)
    y_train = log_returns(train["Close"]).shift(-1).rename("y_next_ret")
    y_test  = log_returns(test["Close"]).shift(-1).rename("y_next_ret")

    # Optionally drop rows with NaN from rolling windows / last label row
    if drop_na_after_features:
        # Align and drop NaNs jointly to keep rows consistent
        data_train = pd.concat([feats_train, y_train], axis=1).dropna()
        data_test  = pd.concat([feats_test,  y_test],  axis=1).dropna()
        feats_train, y_train = data_train.drop(columns=["y_next_ret"]), data_train["y_next_ret"]
        feats_test,  y_test  = data_test.drop(columns=["y_next_ret"]),  data_test["y_next_ret"]

    # Standardize using TRAIN stats only
    X_train_scaled, X_test_scaled, scaler = standardize_train_test(feats_train, feats_test)

    # Prepare output dir
    out_dir = os.path.join(out_base, name)
    os.makedirs(out_dir, exist_ok=True)

    # Save
    X_train_scaled.to_csv(os.path.join(out_dir, "X_train.csv"))
    X_test_scaled.to_csv(os.path.join(out_dir, "X_test.csv"))
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"))
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"))
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump(scaler, f, indent=2)

    print(f"✅ Processed {name} -> {out_dir}")
    print(f"   X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")

# ---------- Convenience: process multiple assets ----------

def process_all(names, in_base="datasets", out_base="processed"):
    for n in names:
        process_asset(n, in_base=in_base, out_base=out_base)

def main():
    reset()
    ds_names = ["snp_500",
                "ftse_100",
                "n_225",
                "eem",
                "gold",
                "tnx"]
    
    process_all(ds_names,"datasets","processed")


if __name__ == "__main__":
    main()