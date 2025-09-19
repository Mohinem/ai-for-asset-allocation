# /scripts/train_ridge.py
import argparse, os, joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data.windows import load_windows
from models.baselines import RidgeBaseline
from scripts.utils import set_seed
import math

def directional_accuracy(yhat: np.ndarray, y: np.ndarray) -> float:
    return np.mean((np.sign(yhat) == np.sign(y)).mean(axis=1))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="ret_1d,ma5_ret,ma21_ret,vol21,rsi14")
    p.add_argument("--mode", choices=["last","flat"], default="last")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--processed_dir", default="processed")
    p.add_argument("--out", default="runs/ridge")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(67)

    Xtr, ytr, Xva, yva, Xte, yte = load_windows(seq_len=60, features=args.features.split(","), processed_dir=args.processed_dir)

    model = RidgeBaseline(alpha=args.alpha).fit(
        np.concatenate([Xtr, Xva], axis=0),
        np.concatenate([ytr, yva], axis=0),
    )
    yhat = model.predict(Xte)

    mse = mean_squared_error(yte, yhat)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(yte, yhat)
    dacc = directional_accuracy(yhat, yte)
    print(f"[Ridge TEST] rmse={rmse:.6f} | mse={mse:.6f} | mae={mae:.6f} | dir_acc={dacc:.4f}")

    joblib.dump(model, os.path.join(args.out, "ridge.joblib"))
