# /scripts/utils.py
import json, os, random
from dataclasses import asdict
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import math

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = float("inf"); self.count = 0
    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value; self.count = 0; return False
        self.count += 1
        return self.count > self.patience

def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    batch_size: int = 32, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def to_tensor(x): return torch.tensor(x, dtype=torch.float32)
    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    val_ds   = TensorDataset(to_tensor(X_val),   to_tensor(y_val))
    test_ds  = TensorDataset(to_tensor(X_test),  to_tensor(y_test))
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ld, val_ld, test_ld

@torch.no_grad()
def evaluate(model, loader, device, eps: float = 1e-500):
    model.eval()
    mse = mae = 0.0
    correct = 0.0
    covered = 0.0
    n = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yhat = model(X)

        diff = (yhat - y)
        mse += torch.mean(diff**2).item() * X.size(0)
        mae += torch.mean(torch.abs(diff)).item() * X.size(0)

        # Directional accuracy with epsilon margin
        mask = (y.abs() > eps) & (yhat.abs() > eps)         # (B, A)
        hits = ((yhat * y) > 0) & mask                      # (B, A)

        covered += mask.float().sum().item()
        correct += hits.float().sum().item()
        n += X.size(0)

    # averaged regression metrics
    reg_mse = mse / n
    reg_mae = mae / n
    reg_rmse = math.sqrt(reg_mse)

    # directional stats
    if covered > 0:
        dir_acc = correct / covered
        coverage = covered / (len(loader.dataset) * loader.dataset.tensors[1].shape[1])
    else:
        dir_acc = 0.0
        coverage = 0.0

    return {
        "mse": reg_mse,
        "rmse": reg_rmse,
        "mae": reg_mae,
        "dir_acc": dir_acc,
        "coverage": coverage
    }


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def save_config(cfg_obj, path: str, extras: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = asdict(cfg_obj) if hasattr(cfg_obj, "__dataclass_fields__") else dict(cfg_obj)
    payload.update(extras or {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
