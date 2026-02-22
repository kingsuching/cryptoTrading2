"""Simple Transformer with multi-head self-attention for crypto price forecasting."""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)             # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CryptoTransformer(nn.Module):
    """Transformer encoder for multivariate crypto price prediction.

    Input  : (batch, seq_len, num_features)
    Output : (batch, 1)

    Parameters
    ----------
    num_features        : number of input features per timestep
    d_model             : embedding dimension
    nhead               : number of attention heads  (must divide d_model)
    num_encoder_layers  : depth of the encoder stack
    dim_feedforward     : inner dimension of the FF sub-layer
    dropout             : dropout rate
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        x = self.dropout(x[:, -1, :])    # last timestep
        return self.fc_out(x)            # (batch, 1)
