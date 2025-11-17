from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.typing import Adj
from einops import rearrange

from tsl.nn.layers import GraphConv
from tsl.nn.models import BaseModel


class STGCN_LSTM(BaseModel):
    """Graph-then-Time model: STGCN -> LSTM -> FC.

    Matches the paper schematic: apply spatial graph convolutions on each
    time slice, then an LSTM across the look-back window for each node,
    followed by a fully-connected layer that outputs the multi-step horizon.

    Notes:
    - Deterministic readout (no sampling head). Use MAE/MSE losses.
    - Exogenous covariates `u` (time) and static features `v` (nodes)
      are concatenated to inputs before spatial layers.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int | None = None,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 64,           # width after spatial graph convs
        graph_layers: int = 2,           # number of spatial layers (STGCN)
        lstm_hidden_size: int = 64,      # LSTM hidden size
        lstm_layers: int = 1,            # LSTM depth
        activation: str = "relu",
        norm: str = "sym",
        root_weight: bool = True,
        cached: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.output_size = output_size or input_size
        self.exog_size = exog_size

        act = activation

        in_ch = input_size + exog_size
        layers = []
        for i in range(graph_layers):
            layers.append(
                GraphConv(
                    in_ch if i == 0 else hidden_size,
                    hidden_size,
                    norm=norm,
                    root_weight=root_weight,
                    cached=cached,
                    activation=act,
                )
            )
        self.spatial = nn.ModuleList(layers)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.readout = nn.Linear(lstm_hidden_size, self.horizon * self.output_size)

    def _concat_exog(self, x: Tensor, u: Optional[Tensor], v: Optional[Tensor]) -> Tensor:
        # x: [B, T, N, F]
        b, t, n, _ = x.shape
        parts = [x]
        if u is not None:
            # u can be [B, T, F] or [B, T, N, F]
            if u.dim() == 3:
                u = u.unsqueeze(2).expand(b, t, n, -1)
            parts.append(u)
        if v is not None:
            # v can be [N, F] or [B, N, F]
            if v.dim() == 2:
                v = v.unsqueeze(0).unsqueeze(0).expand(b, t, n, -1)
            elif v.dim() == 3:
                v = v.unsqueeze(1).expand(b, t, n, -1)
            parts.append(v)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        node_idx: Optional[Tensor] = None,
        mc_samples: Optional[int] = None,
    ) -> Tensor:
        # Expect x: [B, T, N, F]
        x = self._concat_exog(x, u, v)

        # Spatial graph conv per time slice: treat time as batch.
        b, t, n, f = x.shape
        x = rearrange(x, 'b t n f -> (b t) n f')
        for layer in self.spatial:
            x = layer(x, edge_index, edge_weight)
        x = rearrange(x, '(b t) n f -> b t n f', b=b, t=t)

        # Temporal LSTM per node
        x = rearrange(x, 'b t n f -> (b n) t f')
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]  # [B*N, lstm_hidden_size]

        # FC readout to horizon x output_size
        y = self.readout(h_last)  # [B*N, H*out]
        y = y.view(b, n, self.horizon, self.output_size)
        y = rearrange(y, 'b n h f -> b h n f')
        return y
