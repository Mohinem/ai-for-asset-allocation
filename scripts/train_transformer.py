# /scripts/train_transformer.py (updated: sign-focused losses + threshold-robust training)
import argparse, os, math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LambdaLR

from data.windows import load_windows
from models.transformers import TimeSeriesTransformer as Transformer, TransformerConfig
from scripts.utils import set_seed, EarlyStopper, make_loaders, evaluate, save_checkpoint, save_config

def optimize_dir_threshold(model, loader, device, n_samples=8):
    """Finds a prediction magnitude threshold τ that maximizes directional accuracy on validation.
    We keep RMSE calculations on raw predictions; τ only affects direction metric.
    """
    import torch, numpy as np
    model.eval()
    ys, yhats = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            yhat = model.predict_mc(X, n_samples=n_samples) if hasattr(model, "predict_mc") else model(X)
            ys.append(y.cpu())
            yhats.append(yhat.cpu())
    y = torch.cat(ys, 0).numpy()
    yhat = torch.cat(yhats, 0).numpy()

    # Candidate thresholds from percentiles of |yhat|
    mags = np.abs(yhat).reshape(-1)
    qs = np.quantile(mags, np.linspace(0.0, 0.3, 16))  # up to 30th percentile
    best_tau, best_acc = 0.0, -1.0
    for tau in qs:
        s_true = np.sign(y)
        s_pred = np.sign(yhat)
        s_pred[np.abs(yhat) < tau] = 0.0
        acc = (s_true == s_pred).mean()
        if acc > best_acc:
            best_acc, best_tau = acc, float(tau)
    return best_tau, best_acc


def build_scheduler(optim, scheduler_name, epochs, steps_per_epoch, base_lr, warmup_steps=0):
    if scheduler_name == "onecycle":
        return OneCycleLR(
            optim,
            max_lr=base_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100,
            anneal_strategy="cos"
        ), "batch"
    elif scheduler_name == "cosine":
        if warmup_steps > 0:
            def lr_lambda(step):
                return min(1.0, (step + 1) / warmup_steps)
            warmup = LambdaLR(optim, lr_lambda)
        else:
            warmup = None
        cosine = CosineAnnealingLR(optim, T_max=epochs)
        return (warmup, cosine), "hybrid"
    else:
        return None, None

def _metrics_from_lists(y_true_list, y_pred_list):
    import numpy as np
    y_true = np.concatenate([t.cpu().numpy() for t in y_true_list], axis=0)
    y_pred = np.concatenate([t.cpu().numpy() for t in y_pred_list], axis=0)
    mse = float(((y_true - y_pred) ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(y_true - y_pred).mean())
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
    return {"mse": mse, "rmse": rmse, "mae": mae, "dir_acc": dir_acc, "coverage": 1.0}

def evaluate_mc(model, loader, device, n_samples=16):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y_hat = model.predict_mc(X, n_samples=n_samples) if hasattr(model, 'predict_mc') else model(X)
            preds.append(y_hat.detach().cpu())
            trues.append(y.detach().cpu())
    return _metrics_from_lists(trues, preds)

def train(args):
    set_seed(args.seed if not args.no_deterministic else None)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    Xtr, ytr, Xva, yva, Xte, yte = load_windows(seq_len=args.seq_len, features=args.features.split(","))
    n_assets = ytr.shape[1]; input_dim = Xtr.shape[2]

    cfg = TransformerConfig(input_dim=input_dim, output_dim=n_assets,
                            d_model=64, n_heads=8, num_layers=2,
                            dim_feedforward=128, dropout=0.1, weight_decay=1e-4)
    model = Transformer(cfg).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    train_ld, val_ld, test_ld = make_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=args.batch_size)

    steps_per_epoch = max(1, len(train_ld))
    scheduler, sched_mode = build_scheduler(
        optim, args.scheduler, args.epochs, steps_per_epoch, base_lr=args.lr, warmup_steps=args.warmup_steps
    )
    best_val = float("inf")

    use_amp = (device.type == "cuda" and not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0; n = 0

        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                try:
                    yhat, logits = model(X, return_logits=True)
                except TypeError:
                    yhat = model(X)
                    logits = yhat  # proxy logits when aux head isn't present

                # --- Sign-focused multi-objective ---
                # 1) MSE (for RMSE)
                mse = F.mse_loss(yhat, y)

                # 2) BCE on sign with temperature and class imbalance handling
                y_bin = (y >= 0).float()
                pos = y_bin.sum() + 1.0
                neg = y_bin.numel() - y_bin.sum() + 1.0
                pos_weight = neg / pos
                bce = F.binary_cross_entropy_with_logits(args.logit_temp * logits, y_bin, pos_weight=pos_weight)

                # 3) Margin hinge to push same-sign with margin (stabilizes decisions near 0)
                # hinge on product y_true*y_pred
                hinge = F.relu(args.sign_margin - (y * yhat)).mean()

                loss = mse + args.lambda_sign * bce + args.lambda_hinge * hinge

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            if scheduler is not None:
                if args.scheduler == "onecycle":
                    scheduler.step()
                elif args.scheduler == "cosine":
                    warmup, cosine = scheduler
                    if warmup is not None and global_step < args.warmup_steps:
                        warmup.step()

            epoch_loss += loss.item() * X.size(0)
            n += X.size(0)
            global_step += 1

        if scheduler is not None and args.scheduler == "cosine":
            warmup, cosine = scheduler
            cosine.step()

        val_metrics = evaluate(model, val_ld, device)
        mean_train = epoch_loss / max(1, n)
        print(f"[{epoch:03d}] train_mse={mean_train:.6f} | val_mse={val_metrics['mse']:.6f} "
              f"| val_mae={val_metrics['mae']:.6f} | val_dir={val_metrics['dir_acc']:.6f} "
              f"| cov={val_metrics.get('coverage', 0.0):.3f} | lr={optim.param_groups[0]['lr']:.6e}")

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            save_checkpoint(model, os.path.join(args.out, "transformer_best.pt"))


    model.load_state_dict(torch.load(os.path.join(args.out, "transformer_best.pt"), map_location=device))
    test_metrics_std = evaluate(model, test_ld, device)
    test_metrics_mc = evaluate_mc(model, test_ld, device, n_samples=args.mc_samples)
    # Optimize sign threshold on validation for dir_acc
    tau, va_dir = optimize_dir_threshold(model, val_ld, device, n_samples=max(8, args.mc_samples//2))
    print(f"[VAL] best_dir_tau={tau:.6f} | dir_acc@val={va_dir:.4f}")
    print(f"[TEST/STD] rmse={test_metrics_std['rmse']:.6f} | mse={test_metrics_std['mse']:.6f} | mae={test_metrics_std['mae']:.6f} | dir_acc={test_metrics_std['dir_acc']:.4f}")
    print(f"[TEST/MC ] rmse={test_metrics_mc['rmse']:.6f} | mse={test_metrics_mc['mse']:.6f} | mae={test_metrics_mc['mae']:.6f} | dir_acc={test_metrics_mc['dir_acc']:.4f} | dir_tau={tau:.6f}")

    save_config(cfg, os.path.join(args.out, "transformer_config.json"),
                {"epochs": args.epochs, "seed": args.seed, "split": args.split, "features": args.features,
                 "scheduler": args.scheduler, "warmup_steps": args.warmup_steps, "batch_size": args.batch_size,
                 "lr": args.lr, "lambda_sign": args.lambda_sign, "lambda_hinge": args.lambda_hinge,
                 "logit_temp": args.logit_temp, "mc_samples": args.mc_samples,
                 "no_amp": args.no_amp, "no_deterministic": args.no_deterministic,
                 "test_metrics_std": test_metrics_std, "test_metrics_mc": test_metrics_mc})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="2010-2016|2017|2018-2020")
    parser.add_argument("--features", default="ret_1d,ma5_ret,ma21_ret,vol21,rsi14")
    parser.add_argument("--seq_len", type=int, default=60)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["onecycle", "cosine", "none"], default="onecycle")
    parser.add_argument("--warmup_steps", type=int, default=500)

    parser.add_argument("--lambda_sign", type=float, default=0.50, help="weight for sign BCE")
    parser.add_argument("--lambda_hinge", type=float, default=0.20, help="weight for sign-margin hinge")
    parser.add_argument("--sign_margin", type=float, default=0.0015, help="desired margin on y_true*y_pred")
    parser.add_argument("--logit_temp", type=float, default=1.5, help="temperature multiplier for BCE logits")
    parser.add_argument("--mc_samples", type=int, default=16, help="MC-dropout samples for test-time averaging")

    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true", help="disable mixed precision even on CUDA")
    parser.add_argument("--no_deterministic", action="store_true", help="allow nondeterministic/cuDNN autotune")
    parser.add_argument("--out", default="runs/transformer")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)