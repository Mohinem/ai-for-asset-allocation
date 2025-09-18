# /models/transformer.py
from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass
class TransformerConfig:
    input_dim: int              # n_assets * n_features
    output_dim: int             # n_assets
    d_model: int = 64           # paper: 64
    n_heads: int = 8            # paper: 8
    num_layers: int = 2         # paper: 2 encoder blocks
    dim_feedforward: int = 128  # paper: 128
    dropout: float = 0.1        # paper: 0.1
    max_len: int = 512          # >= 60
    weight_decay: float = 1e-4  # paper: 1e-4
    pool: str = "mean"          # NEW: "last" | "mean" | "cls"

class PositionalEncoding(nn.Module):
    """Standard sinusoidal PE (no learnable params)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """
    Encoder-only Transformer for multi-output next-day return prediction.
    Paper hyperparameters are preserved:
      - d_model=64, n_heads=8, num_layers=2, dim_feedforward=128, dropout=0.1
      - Sinusoidal positional encoding
      - Linear head -> output_dim (n_assets)
      - Loss: multi-output MSE
    Improvements (do not alter paper config):
      - LayerNorm after input projection
      - Flexible pooling (last | mean | cls)
      - Pre-head LayerNorm for stability
    Input:  (B, T=60, F=input_dim)
    Output: (B, output_dim)
    """
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.in_norm = nn.LayerNorm(cfg.d_model)           # NEW

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.posenc = PositionalEncoding(cfg.d_model, max_len=cfg.max_len, dropout=0.0)

        # Optional CLS token for pooling
        if cfg.pool == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        else:
            self.register_parameter("cls", None)

        self.pre_head_norm = nn.LayerNorm(cfg.d_model)     # NEW
        self.head = nn.Linear(cfg.d_model, cfg.output_dim)

        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight);       nn.init.zeros_(self.head.bias)

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.pool == "last":
            return z[:, -1, :]         # (B, d_model)
        if self.cfg.pool == "mean":
            return z.mean(dim=1)       # (B, d_model)
        # cls
        return z[:, 0, :]              # (B, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        z = self.input_proj(x)         # (B, T, d_model)
        z = self.in_norm(z)            # NEW: stabilize scales
        z = self.posenc(z)

        if self.cfg.pool == "cls":
            B = z.size(0)
            cls = self.cls.expand(B, -1, -1)   # (B,1,d_model)
            z = torch.cat([cls, z], dim=1)     # prepend token

        z = self.encoder(z)             # (B, T[, +1], d_model)
        pooled = self._pool(z)          # (B, d_model)
        pooled = self.pre_head_norm(pooled)
        y = self.head(pooled)           # (B, output_dim)
        return y
