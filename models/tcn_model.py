"""
Temporal Convolutional Network (TCN) for Trajectory Fragment Stitching

Architecture:
  Input: (batch, seq_len, 8) raw features
  -> TemporalBlock stack with dilated causal convolutions + residual connections
  -> AdaptiveAvgPool || AdaptiveMaxPool -> concat -> embedding
  -> Similarity head with endpoint features -> probability

~80K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalBlock(nn.Module):
    """
    Single TCN block: dilated causal conv + residual connection.

    Conv1d -> BatchNorm -> ReLU -> Dropout -> Conv1d -> BatchNorm -> ReLU -> Dropout
    + residual (with 1x1 conv if channel sizes differ)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        res = self.residual(x)

        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # causal trim
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # causal trim
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        return F.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Stack of TemporalBlocks producing fixed-size embeddings via pooling.

    Architecture:
        TemporalBlock(8->32, k=3, d=1)
        TemporalBlock(32->64, k=3, d=2)
        TemporalBlock(64->64, k=3, d=4)
        AdaptiveAvgPool1d(1) || AdaptiveMaxPool1d(1) -> concat -> 128-dim
        LayerNorm(128)
    """

    def __init__(self, input_size: int = 8, channels: tuple = (32, 64, 64),
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()

        layers = []
        in_ch = input_size
        dilation = 1
        for out_ch in channels:
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
            dilation *= 2
        self.tcn = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.output_size = channels[-1] * 2  # avg + max concat
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, output_size) embedding
        """
        # TCN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)

        avg = self.avg_pool(x).squeeze(-1)
        mx = self.max_pool(x).squeeze(-1)
        emb = torch.cat([avg, mx], dim=1)
        return self.layer_norm(emb)


class SiameseTCN(nn.Module):
    """
    Siamese TCN for trajectory fragment similarity.

    Twin TCN encoders (shared) -> embeddings -> similarity head with endpoint features.
    """

    def __init__(self, input_size: int = 8, channels: tuple = (32, 64, 64),
                 kernel_size: int = 3, dropout: float = 0.2,
                 endpoint_dim: int = 4):
        super().__init__()

        self.encoder = TCNEncoder(input_size, channels, kernel_size, dropout)
        emb_size = self.encoder.output_size  # 128

        # Similarity head: [emb_a; emb_b; endpoint_feats] -> probability
        combined_dim = emb_size * 2 + endpoint_dim  # 128*2 + 4 = 260
        self.similarity_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor,
                endpoint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_a: (batch, seq_len_a, 8)
            seq_b: (batch, seq_len_b, 8)
            endpoint_features: (batch, 4) gap features
        Returns:
            (batch, 1) similarity probability
        """
        emb_a = self.encoder(seq_a)
        emb_b = self.encoder(seq_b)
        combined = torch.cat([emb_a, emb_b, endpoint_features], dim=1)
        return self.similarity_head(combined)

    def get_embeddings(self, seq_a: torch.Tensor, seq_b: torch.Tensor):
        """Get embeddings without computing similarity (for contrastive loss)."""
        return self.encoder(seq_a), self.encoder(seq_b)


if __name__ == "__main__":
    model = SiameseTCN()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TCN parameters: {n_params:,}")
    print(model)

    # Test forward pass
    seq_a = torch.randn(4, 20, 8)
    seq_b = torch.randn(4, 15, 8)
    ep = torch.randn(4, 4)
    out = model(seq_a, seq_b, ep)
    print(f"Input: A={seq_a.shape}, B={seq_b.shape} -> Output: {out.shape}")
    print(f"Output: {out.squeeze().detach()}")
