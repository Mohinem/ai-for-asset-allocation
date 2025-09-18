# /models/baselines.py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Literal

class RidgeBaseline:
    """
    Linear baseline on either:
      - mode='last': last timestep features only
      - mode='flat': flattened (T * F) window
    Target: multi-asset next-day returns (shape (n_assets,))
    """
    def __init__(self, alpha: float = 1.0, mode: Literal["last", "flat"] = "last"):
        self.mode = mode
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=0))
        ])

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        # X: (N, T, F)
        if self.mode == "last":
            return X[:, -1, :]                  # (N, F)
        elif self.mode == "flat":
            N, T, F = X.shape
            return X.reshape(N, T * F)          # (N, T*F)
        else:
            raise ValueError("mode must be 'last' or 'flat'")

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xp = self._reshape(X)
        self.model.fit(Xp, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xp = self._reshape(X)
        return self.model.predict(Xp)

def naive_zero(N: int, n_assets: int) -> np.ndarray:
    """Predict zero return for all assets."""
    return np.zeros((N, n_assets), dtype=float)

def naive_persistence(prev_returns: np.ndarray) -> np.ndarray:
    """
    Predict y_hat_{t+1} = r_t  (persistence / random-walk).
    prev_returns: shape (N, n_assets) — the last observed returns.
    """
    return prev_returns.copy()
