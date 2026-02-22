"""Multi-layer LSTM model using PyTorch for sequential price forecasting."""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Multi-layer LSTM for sequence-to-one price prediction.

    Expects input of shape  (batch_size, seq_len, input_size).
    Returns shape           (batch_size, 1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)            # (batch, seq_len, hidden)
        out = self.dropout(out[:, -1, :])  # last timestep
        return self.fc(out)              # (batch, output_size)
