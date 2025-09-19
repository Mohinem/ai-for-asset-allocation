# /models/lstm.py
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class LSTMConfig:
    input_dim: int              # n_assets * n_features
    output_dim: int             # n_assets
    hidden_size: int = 50       # paper: 2 LSTM layers, 50 units each
    num_layers: int = 2
    dropout: float = 0.2        # between LSTM layers
    bidirectional: bool = False # standard (not bi-LSTM)

class LSTM(nn.Module):
    """
    Multi-output LSTM model for next-day return prediction.
    Paper hyperparameters:
      - 2 LSTM layers (50 units), dropout=0.2
      - Linear/MLP head -> output_dim (n_assets)
      - Loss: multi-output MSE
    Input:  (B, T=60, F=input_dim)
    Output: (B, output_dim)
    """
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        # Input stabilization
        self.in_norm = nn.LayerNorm(cfg.input_dim)

        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )

        head_in = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        # Slightly stronger head (helps without changing LSTM capacity)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_in, cfg.output_dim),
        )

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stabilize
        x = self.in_norm(x)
        # Run LSTM
        out, _ = self.lstm(x)         # (B, T, H)
        # Mean-pool across time (empirically better than last-only for noisy returns)
        pooled = out.mean(dim=1)      # (B, H)
        y = self.head(pooled)         # (B, output_dim)
        return y
