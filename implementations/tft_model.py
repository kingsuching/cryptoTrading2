"""Temporal Fusion Transformer (TFT) implementation using PyTorch.

Implements key TFT components:
  - Gated Linear Unit (GLU)
  - Gated Residual Network (GRN)
  - Variable Selection Network (VSN)
  - LSTM encoder with post-LSTM gating
  - Temporal multi-head self-attention
  - Position-wise feed-forward (via GRN)
"""
import math
import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """GLU: splits linear projection in two halves, gates the second."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN: ELU non-linearity + GLU gating + skip connection + layer norm."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None
        self.fc2 = nn.Linear(hidden_size, output_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        h = nn.functional.elu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        a, b = h.chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        residual = self.skip(x) if self.skip is not None else x
        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: soft-selects the most relevant features at each timestep."""

    def __init__(
        self,
        num_variables: int,
        variable_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.grns = nn.ModuleList(
            [GatedResidualNetwork(variable_size, hidden_size, hidden_size, dropout)
             for _ in range(num_variables)]
        )
        self.selection_grn = GatedResidualNetwork(
            num_variables * variable_size, hidden_size, num_variables, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """x: (..., num_variables, variable_size)"""
        batch_shape = x.shape[:-2]
        flat = x.reshape(*batch_shape, -1)
        weights = self.softmax(self.selection_grn(flat))          # (..., n_vars)
        processed = torch.stack(
            [self.grns[i](x[..., i, :]) for i in range(self.num_variables)], dim=-2
        )                                                           # (..., n_vars, hidden)
        output = (weights.unsqueeze(-1) * processed).sum(dim=-2)  # (..., hidden)
        return output, weights


class TemporalFusionTransformer(nn.Module):
    """Simplified TFT for multivariate crypto price forecasting.

    Parameters
    ----------
    num_features      : number of input features per timestep
    hidden_size       : hidden dimension throughout the model
    num_heads         : attention heads
    num_lstm_layers   : depth of LSTM encoder
    dropout           : dropout rate
    forecast_horizon  : number of future timesteps to predict (default 1)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 1,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        # Project each scalar feature to hidden_size
        self.input_proj = nn.Linear(1, hidden_size)

        # Variable selection network
        self.vsn = VariableSelectionNetwork(num_features, hidden_size, hidden_size, dropout)

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Post-LSTM gating + norm
        self.post_lstm_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        # Temporal self-attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.post_attn_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # Position-wise FF (GRN) + norm
        self.ff_grn = GatedResidualNetwork(hidden_size, hidden_size * 4, hidden_size, dropout)
        self.ff_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.output_fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x : (batch, seq_len, num_features)
        Returns
            (batch, forecast_horizon)
        """
        B, T, F = x.shape
        # Project each feature to hidden_size: (B, T, F, hidden)
        x_proj = self.input_proj(x.unsqueeze(-1))

        # Variable selection per timestep
        vsn_out = torch.stack(
            [self.vsn(x_proj[:, t, :, :])[0] for t in range(T)], dim=1
        )  # (B, T, hidden)

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(vsn_out)
        lstm_gated = self.post_lstm_gate(lstm_out)
        lstm_out = self.post_lstm_norm(lstm_gated + vsn_out)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_gated = self.post_attn_gate(attn_out)
        attn_out = self.post_attn_norm(attn_gated + lstm_out)

        # Feed-forward
        ff_out = self.ff_grn(attn_out)
        ff_out = self.ff_norm(ff_out + attn_out)

        # Decode from last timestep
        last = self.dropout(ff_out[:, -1, :])
        return self.output_fc(last)  # (B, forecast_horizon)
