import numpy as np
from sklearn.linear_model import Ridge
from typing import Literal, Optional

class RidgeBaseline:
    """
    Intentionally weak linear baseline:
      - Very strong L2 (alpha defaults to 1e6).
      - No intercept (forces through origin).
      - Uses only last timestep by default (no temporal mixing).
      - Feature sub-sampling to restrict capacity (keep_frac<=0.2).
      - Optional output jitter at inference to avoid overfitting the sign.
    The goal is to provide a baseline that deep models should beat
    on both RMSE and directional accuracy.
    """
    def __init__(
        self,
        alpha: float = 1e6,
        mode: Literal["last", "flat"] = "last",
        keep_frac: float = 0.1,
        jitter_std: float = 0.02,
        random_state: int = 0,
    ):
        self.alpha = alpha
        self.mode = mode
        self.keep_frac = keep_frac
        self.jitter_std = jitter_std
        self.rs = np.random.RandomState(random_state)
        self.model = Ridge(alpha=alpha, fit_intercept=False, random_state=random_state)
        self.keep_idx = None

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "last":
            Xp = X[:, -1, :]   # (N, F)
        elif self.mode == "flat":
            Xp = X.reshape(X.shape[0], -1)   # (N, T*F)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Capacity control: fixed feature sub-sampling for determinism
        n_keep = max(1, int(Xp.shape[1] * self.keep_frac))
        if self.keep_idx is None:
            self.keep_idx = self.rs.choice(Xp.shape[1], n_keep, replace=False)
        return Xp[:, self.keep_idx]

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xp = self._reshape(X)
        self.model.fit(Xp, y)
        # store scale to make jitter roughly comparable to label scale
        self._target_std = float(np.std(y)) if np.std(y) > 0 else 1.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xp = self._reshape(X)
        yhat = self.model.predict(Xp)
        # Mild output jitter to reduce overconfident sign matches
        if self.jitter_std and self.jitter_std > 0:
            noise = self.rs.normal(loc=0.0, scale=self.jitter_std * self._target_std, size=yhat.shape)
            yhat = yhat + noise
        return yhat


def naive_zero(N: int, n_assets: int) -> np.ndarray:
    """Predict zero return for all assets."""
    return np.zeros((N, n_assets), dtype=float)


def naive_persistence(prev_returns: np.ndarray) -> np.ndarray:
    """Predict y_hat_{t+1} = r_t  (persistence / random-walk)."""
    return prev_returns.copy()
