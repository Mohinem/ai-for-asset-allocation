# /scripts/train_transformer.py
import argparse, os
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LambdaLR

from data.windows import load_windows
from models.transformers import TimeSeriesTransformer as Transformer, TransformerConfig
from scripts.utils import set_seed, EarlyStopper, make_loaders, evaluate, save_checkpoint, save_config

def build_scheduler(optim, scheduler_name, epochs, steps_per_epoch, base_lr, warmup_steps=0):
    if scheduler_name == "onecycle":
        # OneCycle handles its own warmup; step every batch
        return OneCycleLR(
            optim,
            max_lr=base_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.1,        # 10% warmup portion
            div_factor=10.0,      # initial lr = max_lr / 10
            final_div_factor=100, # final lr = max_lr / 100
            anneal_strategy="cos"
        ), "batch"
    elif scheduler_name == "cosine":
        # Linear warmup (LambdaLR), then cosine per-epoch
        if warmup_steps > 0:
            def lr_lambda(step):
                return min(1.0, (step + 1) / warmup_steps)
            warmup = LambdaLR(optim, lr_lambda)
        else:
            warmup = None
        cosine = CosineAnnealingLR(optim, T_max=epochs)
        # We'll handle: warmup stepped per-batch for `warmup_steps`, then cosine per-epoch
        return (warmup, cosine), "hybrid"
    else:
        return None, None

def train(args):
    set_seed(args.seed if not args.no_deterministic else None)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load pre-windowed data
    Xtr, ytr, Xva, yva, Xte, yte = load_windows(
        seq_len=args.seq_len, features=args.features.split(",")
    )
    n_assets = ytr.shape[1]; input_dim = Xtr.shape[2]

    # Model + optimizer + loss
    cfg = TransformerConfig(input_dim=input_dim, output_dim=n_assets,
                            d_model=64, n_heads=8, num_layers=2,
                            dim_feedforward=128, dropout=0.1, weight_decay=1e-4)
    model = Transformer(cfg).to(device)
    criterion = nn.MSELoss()
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    # DataLoaders
    train_ld, val_ld, test_ld = make_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=args.batch_size)

    # Scheduler
    steps_per_epoch = max(1, len(train_ld))
    scheduler, sched_mode = build_scheduler(
        optim, args.scheduler, args.epochs, steps_per_epoch, base_lr=args.lr, warmup_steps=args.warmup_steps
    )

    # Early stopping (a tad more patient for Transformer)
    stopper = EarlyStopper(patience=args.patience, min_delta=0.0)
    best_val = float("inf")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0; n = 0

        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                yhat = model(X)
                loss = criterion(yhat, y)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            # LR schedule stepping
            if scheduler is not None:
                if args.scheduler == "onecycle":
                    scheduler.step()  # per-batch
                elif args.scheduler == "cosine":
                    # hybrid: linear warmup for first warmup_steps *batches*
                    warmup, cosine = scheduler
                    if warmup is not None and global_step < args.warmup_steps:
                        warmup.step()
                # else 'none' -> no step

            epoch_loss += loss.item() * X.size(0)
            n += X.size(0)
            global_step += 1

        # Cosine (epoch step)
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

        if stopper.step(val_metrics["mse"]):
            print("Early stopping triggered."); break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(os.path.join(args.out, "transformer_best.pt"), map_location=device))
    test_metrics = evaluate(model, test_ld, device)
    print(f"[TEST] mse={test_metrics['mse']:.6f} | mae={test_metrics['mae']:.6f} "
          f"| dir_acc={test_metrics['dir_acc']:.6f} | cov={test_metrics.get('coverage', 0.0):.3f}")

    # Save config + run args
    save_config(cfg, os.path.join(args.out, "transformer_config.json"),
                {"epochs": args.epochs, "seed": args.seed, "split": args.split, "features": args.features,
                 "scheduler": args.scheduler, "warmup_steps": args.warmup_steps, "batch_size": args.batch_size,
                 "lr": args.lr, "no_amp": args.no_amp, "no_deterministic": args.no_deterministic,
                 "test_metrics": test_metrics})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="2010-2016|2017|2018-2020")
    parser.add_argument("--features", default="ret_1d,ma5_ret,ma21_ret,vol21,rsi14")
    parser.add_argument("--seq_len", type=int, default=60)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["onecycle", "cosine", "none"], default="onecycle")
    parser.add_argument("--warmup_steps", type=int, default=500)   # used only for cosine/hybrid

    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true", help="disable mixed precision even on CUDA")
    parser.add_argument("--no_deterministic", action="store_true", help="allow nondeterministic/cuDNN autotune")
    parser.add_argument("--out", default="runs/transformer")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)
