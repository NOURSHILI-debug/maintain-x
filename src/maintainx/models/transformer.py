"""Transformer encoder for RUL regression on windowed sequences."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sine/cosine positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class RULTransformer(nn.Module):
    """Encoder-only transformer: window of cycles -> single RUL prediction.

    Args:
        n_features: number of input features per cycle (your ~86 columns).
        d_model: internal embedding dimension.
        n_heads: attention heads per encoder block.
        n_layers: number of stacked encoder blocks.
        d_ff: hidden dim of the feed-forward sublayer.
        dropout: dropout rate.
    """

    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.1):
        super().__init__()

        # Project raw features into d_model space (tokens don't start at d_model width)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # (batch, seq, feature) instead of (seq, batch, feature)
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Regression head: mean-pool over the sequence, then a small MLP -> 1 value
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (batch, window, n_features)
        x = self.input_proj(x)          # (batch, window, d_model)
        x = self.pos_encoding(x)
        x = self.encoder(x)             # (batch, window, d_model) — attention happens here
        x = x.mean(dim=1)               # pool across the window -> (batch, d_model)
        rul = self.head(x).squeeze(-1)  # (batch,)
        return rul