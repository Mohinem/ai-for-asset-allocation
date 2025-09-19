# /scripts/train_lstm.py
import argparse, os
import torch
from torch import nn
from torch.optim import Adam

from data.windows import load_windows   # uses paper split: train(2010–2017), val=last 10% train, test(2018–2020)
from models.lstm import LSTM, LSTMConfig
from scripts.utils import set_seed, EarlyStopper, make_loaders, evaluate, save_checkpoint, save_config

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load windows with paper-consistent splitting
    Xtr, ytr, Xva, yva, Xte, yte = load_windows(
        seq_len=60,
        features=args.features.split(","),
        processed_dir=args.processed_dir,
        val_ratio_in_train=0.10,
    )
    n_assets = ytr.shape[1]; input_dim = Xtr.shape[2]

    cfg = LSTMConfig(input_dim=input_dim, output_dim=n_assets, hidden_size=50, num_layers=2, dropout=0.2)
    model = LSTM(cfg).to(device)

    criterion = nn.MSELoss()                      # multi-output MSE
    optim = Adam(model.parameters(), lr=1e-3)     # paper: Adam 1e-3

    train_ld, val_ld, test_ld = make_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=32)
    stopper = EarlyStopper(patience=1000)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            yhat = model(X)
            loss = criterion(yhat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            running += loss.item() * X.size(0); n += X.size(0)

        val_metrics = evaluate(model, val_ld, device)
        print(f"[{epoch:03d}] train_mse={running/n:.6f} | val_mse={val_metrics['mse']:.6f} "
              f"| val_mae={val_metrics['mae']:.6f} | val_dir={val_metrics['dir_acc']:.4f}")

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            save_checkpoint(model, os.path.join(args.out, "lstm_best.pt"))

        if stopper.step(val_metrics["mse"]):
            print("Early stopping triggered."); break

    model.load_state_dict(torch.load(os.path.join(args.out, "lstm_best.pt"), map_location=device))
    test_metrics = evaluate(model, test_ld, device)
    print(f"[TEST] rmse={test_metrics['rmse']:.6f} | mse={test_metrics['mse']:.6f} | mae={test_metrics['mae']:.6f} | dir_acc={test_metrics['dir_acc']:.4f}")

    save_config(cfg, os.path.join(args.out, "lstm_config.json"),
                {"epochs": args.epochs, "seed": args.seed,
                 "features": args.features, "processed_dir": args.processed_dir,
                 "test_metrics": test_metrics})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="ret_1d,ma5_ret,ma21_ret,vol21,rsi14")
    p.add_argument("--epochs", type=int, default=100)  # paper
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--processed_dir", default="processed")
    p.add_argument("--out", default="runs/lstm")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)
