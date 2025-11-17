import torch
import torch.nn as nn
from typing import Optional

class SimpleGRUBaseline(nn.Module):
    """
    Model0: Pure temporal GRU baseline, no graphs, no embeddings.
    Each node's time series is processed by the same GRU.
    """

    def __init__(
        self,
        input_size: int,          # F
        horizon: int,             # forecast horizon
        n_nodes: int,             # N
        output_size: Optional[int] = None,  # defaults to input_size
        hidden_size: int = 64,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.output_size = output_size or input_size

        rnn_type = rnn_type.lower()
        if rnn_type == "gru":
            rnn_cls = nn.GRU
        elif rnn_type == "lstm":
            rnn_cls = nn.LSTM
        else:
            rnn_cls = nn.RNN

        # Single global RNN shared by all nodes
        self.rnn = rnn_cls(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input will be (B*N, T, F)
        )

        # Map final hidden state -> full horizon for one node
        self.readout = nn.Linear(
            self.hidden_size,
            self.horizon * self.output_size,
        )

    def forward(self, x):
        """
        x: (B, T, N, F)
        returns: (B, horizon, N, output_size)
        """
        B, T, N, F = x.shape
        assert N == self.n_nodes
        assert F == self.input_size

        # Merge batch and nodes: (B*N, T, F)
        x = x.view(B * N, T, F)

        # RNN over time
        rnn_out, _ = self.rnn(x)           # (B*N, T, hidden)
        last_hidden = rnn_out[:, -1, :]    # (B*N, hidden)

        # Predict entire horizon in one shot
        out = self.readout(last_hidden)    # (B*N, horizon * output_size)
        out = out.view(B, N, self.horizon, self.output_size)
        out = out.permute(0, 2, 1, 3)      # (B, horizon, N, output_size)

        return out
