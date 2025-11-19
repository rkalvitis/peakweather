from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.typing import Adj

from tsl.nn.layers import GraphConv
from tsl.nn.models import BaseModel


class _CausalConv1d(nn.Module):
    """1D convolution with manual left padding to keep causality."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class _MultiScaleDilatedBlock(nn.Module):
    """Stack of causal dilated convolutions operating at several scales."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int],
        dilations: Sequence[int],
    ):
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same length.")
        splits = [channels // len(kernel_sizes)] * len(kernel_sizes)
        remainder = channels - sum(splits)
        for i in range(remainder):
            splits[i % len(splits)] += 1
        self.convs = nn.ModuleList(
            _CausalConv1d(
                in_channels=channels,
                out_channels=split,
                kernel_size=k,
                dilation=d,
            )
            for split, k, d in zip(splits, kernel_sizes, dilations)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        return out.transpose(1, 2)


class MultiScaleGatedTCN(nn.Module):
    """Implementation of the multi-scale gated temporal convolution module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        dilations: Sequence[int] = (1, 2, 3, 4),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU())
        self.filter_block = _MultiScaleDilatedBlock(hidden_dim, kernel_sizes, dilations)
        self.gate_block = _MultiScaleDilatedBlock(hidden_dim, kernel_sizes, dilations)
        self.filter_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        h_l = self.filter_fc(self.filter_block(x))
        h_r = self.gate_fc(self.gate_block(x))
        out = torch.tanh(h_l) * torch.sigmoid(h_r)
        return self.dropout(out)


class MultiGranularityRandomGraphAttention(nn.Module):
    """Captures periodic dependencies across hours and days via graph attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        max_segments: int = 64,
        segments_per_day: int = 24,
        days_per_cycle: int = 7,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.segments_per_day = segments_per_day
        self.days_per_cycle = days_per_cycle
        self.max_segments = max_segments

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.random_bias = nn.Parameter(torch.randn(2, max_segments, max_segments) * 0.01)

    @staticmethod
    def _build_hour_adj(length: int, segments_per_day: int, device: torch.device) -> Tensor:
        idx = torch.arange(length, device=device)
        hour_ids = idx % segments_per_day
        adj = (hour_ids[None, :] == hour_ids[:, None]).float()
        return adj

    @staticmethod
    def _build_day_adj(length: int, segments_per_day: int, days_per_cycle: int, device: torch.device) -> Tensor:
        idx = torch.arange(length, device=device)
        day_ids = (idx // segments_per_day) % max(days_per_cycle, 1)
        adj = (day_ids[None, :] == day_ids[:, None]).float()
        return adj

    def _graph_attention(self, q: Tensor, k: Tensor, v: Tensor, adj: Tensor, bias: Tensor) -> Tensor:
        mask = (adj == 0).unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
        bias = bias.unsqueeze(0)  # [1,S,S]

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k) / math.sqrt(self.head_dim)
        scores = scores + bias.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bnqk,bnkd->bnqd", attn, v)
        return out

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, H]
        b, t, _ = x.shape
        device = x.device
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        hour_adj = self._build_hour_adj(t, self.segments_per_day, device)
        day_adj = self._build_day_adj(t, self.segments_per_day, self.days_per_cycle, device)
        hour_bias = self.random_bias[0, :t, :t]
        day_bias = self.random_bias[1, :t, :t]

        hour_out = self._graph_attention(q, k, v, hour_adj, hour_bias)
        day_out = self._graph_attention(q, k, v, day_adj, day_bias)

        hour_out = hour_out.transpose(1, 2).reshape(b, t, self.hidden_dim)
        day_out = day_out.transpose(1, 2).reshape(b, t, self.hidden_dim)
        combined = torch.cat([hour_out, day_out], dim=-1)
        return self.out_proj(combined)


class ShortTermSTGNNEncoder(nn.Module):
    """STGNN-style encoder applied to the most recent window."""

    def __init__(
        self,
        hidden_size: int,
        graph_layers: int = 2,
        time_layers: int = 1,
        activation: str = "relu",
        norm: str = "sym",
        root_weight: bool = True,
        cached: bool = False,
    ):
        super().__init__()
        self.preprocess = nn.Sequential(nn.LazyLinear(hidden_size), nn.ReLU())
        self.graph_layers = nn.ModuleList(
            GraphConv(
                hidden_size,
                hidden_size,
                activation=activation,
                norm=norm,
                root_weight=root_weight,
                cached=cached,
            )
            for _ in range(graph_layers)
        )
        self.temporal = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=time_layers,
            batch_first=True,
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Optional[Tensor] = None) -> Tensor:
        # x: [B, T, N, F]
        b, t, n, _ = x.shape
        x = self.preprocess(x)
        x = x.view(b * t, n, -1)
        for layer in self.graph_layers:
            x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.view(b, t, n, -1)
        x = rearrange(x, "b t n c -> (b n) t c")
        output, _ = self.temporal(x)
        feats = output[:, -1, :]
        return feats.view(b, n, -1)


class TransformerSpatioTemporalGAT(BaseModel):
    """Implementation of the Transformer-based Spatiotemporal Graph Attention Network."""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: Optional[int] = None,
        output_size: Optional[int] = None,
        exog_size: int = 0,
        transformer_d_model: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff_size: int = 256,
        dropout: float = 0.1,
        segment_length: int = 24,
        max_segments: int = 32,
        segments_per_day: int = 24,
        days_per_cycle: int = 7,
        long_term_kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        long_term_dilations: Sequence[int] = (1, 2, 3, 4),
        short_window: int = 24,
        short_hidden_size: int = 64,
        short_graph_layers: int = 2,
        short_time_layers: int = 1,
        graph_activation: str = "relu",
        graph_norm: str = "sym",
        graph_root_weight: bool = True,
        graph_cached: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.exog_size = exog_size
        self.segment_length = segment_length
        self.max_segments = max_segments
        self.short_window = short_window
        self.transformer_d_model = transformer_d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.segment_embed = nn.LazyLinear(transformer_d_model)
        self.positional_dropout = nn.Dropout(dropout)

        self.long_term = MultiScaleGatedTCN(
            input_dim=transformer_d_model,
            hidden_dim=transformer_d_model,
            kernel_sizes=long_term_kernel_sizes,
            dilations=long_term_dilations,
            dropout=dropout,
        )
        self.periodic = MultiGranularityRandomGraphAttention(
            hidden_dim=transformer_d_model,
            num_heads=transformer_heads,
            max_segments=max_segments,
            segments_per_day=segments_per_day,
            days_per_cycle=days_per_cycle,
        )
        self.short_term = ShortTermSTGNNEncoder(
            hidden_size=short_hidden_size,
            graph_layers=short_graph_layers,
            time_layers=short_time_layers,
            activation=graph_activation,
            norm=graph_norm,
            root_weight=graph_root_weight,
            cached=graph_cached,
        )

        fusion_dim = transformer_d_model * 2 + short_hidden_size
        self.readout = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, horizon * self.output_size),
        )

    @staticmethod
    def _positional_encoding(length: int, dim: int, device: torch.device) -> Tensor:
        pe = torch.zeros(length, dim, device=device)
        position = torch.arange(0, length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def _merge_features(x: Tensor, u: Optional[Tensor]) -> Tensor:
        if u is None:
            return x
        if u.dim() == 3:
            u = u.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        elif u.dim() == 4 and u.size(2) == 1:
            u = u.expand(-1, -1, x.size(2), -1)
        return torch.cat([x, u], dim=-1)

    def _reshape_segments(self, x: Tensor) -> tuple[Tensor, int]:
        b, t, n, f = x.shape
        seg_len = self.segment_length
        if seg_len <= 0:
            raise ValueError("segment_length must be positive.")
        num_segments = math.ceil(t / seg_len)
        num_segments = min(num_segments, self.max_segments)
        target_len = num_segments * seg_len
        if t < target_len:
            pad = torch.zeros(b, target_len - t, n, f, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
        elif t > target_len:
            x = x[:, -target_len:, :, :]

        x = x.view(b, num_segments, seg_len, n, f)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        bn = b * n
        segments = x.view(bn, num_segments, seg_len * f)
        return segments, num_segments

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
        _ = v  # Static attributes are merged into u upstream.
        _ = mc_samples

        x = self._merge_features(x, u)
        b, t, n, f = x.shape
        segments, num_segments = self._reshape_segments(x)

        pos = self._positional_encoding(num_segments, self.transformer_d_model, x.device)
        tokens = self.segment_embed(segments) + pos
        tokens = self.positional_dropout(tokens)
        tokens = self.transformer(tokens)

        long_seq = self.long_term(tokens)
        periodic_seq = self.periodic(tokens)
        long_feat = long_seq[:, -1, :].view(b, n, -1)
        periodic_feat = periodic_seq[:, -1, :].view(b, n, -1)

        short_window = min(self.short_window, t)
        short_seq = x[:, -short_window:, :, :]
        short_feat = self.short_term(short_seq, edge_index=edge_index, edge_weight=edge_weight)

        features = torch.cat([long_feat, periodic_feat, short_feat], dim=-1)
        if node_idx is not None:
            features = features[:, node_idx, :]
        out = self.readout(features.view(-1, features.size(-1)))
        out = out.view(features.size(0), features.size(1), self.horizon, self.output_size)
        out = out.permute(0, 2, 1, 3)
        return out
