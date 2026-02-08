"""
Siamese Transformer for Trajectory Fragment Stitching

Architecture:
  Input: (batch, seq_len, 8) raw features
  -> Linear projection (8 -> 64)
  -> Sinusoidal positional encoding
  -> TransformerEncoder (2 layers, 4 heads, d_ff=128)
  -> Masked mean pooling -> 64-dim embedding
  -> Similarity head with |emb_a - emb_b| + endpoint features -> probability

~55K parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTrajectoryEncoder(nn.Module):
    """
    Transformer encoder for variable-length trajectory sequences.

    Linear(8, 64) -> PositionalEncoding -> TransformerEncoder -> masked mean pool
    """

    def __init__(self, input_size: int = 8, d_model: int = 64, nhead: int = 4,
                 dim_feedforward: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_size = d_model  # 64

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            padding_mask: (batch, seq_len) True = padded positions to ignore
        Returns:
            (batch, d_model) embedding via masked mean pooling
        """
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Masked mean pooling
        if padding_mask is not None:
            # Invert mask: True -> valid positions
            valid_mask = ~padding_mask  # (batch, seq_len)
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            x_masked = x * valid_mask_expanded
            lengths = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (batch, 1)
            pooled = x_masked.sum(dim=1) / lengths  # (batch, d_model)
        else:
            pooled = x.mean(dim=1)

        return self.layer_norm(pooled)


class SiameseTransformerNetwork(nn.Module):
    """
    Siamese Transformer for trajectory fragment similarity.

    Twin encoders (shared) -> [emb_a; emb_b; |emb_a - emb_b|; endpoint_feats]
    -> similarity head -> probability

    Key difference from TCN: adds |emb_a - emb_b| element-wise difference.
    """

    def __init__(self, input_size: int = 8, d_model: int = 64, nhead: int = 4,
                 dim_feedforward: int = 128, num_layers: int = 2, dropout: float = 0.2,
                 endpoint_dim: int = 4):
        super().__init__()

        self.encoder = TransformerTrajectoryEncoder(
            input_size=input_size, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=num_layers, dropout=dropout,
        )

        emb_size = self.encoder.output_size  # 64
        # [emb_a; emb_b; |emb_a - emb_b|; endpoint_feats] = 64 + 64 + 64 + 4 = 196
        combined_dim = emb_size * 3 + endpoint_dim

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

    def forward(self, seq_a: torch.Tensor, mask_a: Optional[torch.Tensor],
                seq_b: torch.Tensor, mask_b: Optional[torch.Tensor],
                endpoint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_a: (batch, seq_len_a, 8)
            mask_a: (batch, seq_len_a) padding mask (True = pad)
            seq_b: (batch, seq_len_b, 8)
            mask_b: (batch, seq_len_b) padding mask
            endpoint_features: (batch, 4)
        Returns:
            (batch, 1) similarity probability
        """
        emb_a = self.encoder(seq_a, mask_a)
        emb_b = self.encoder(seq_b, mask_b)
        diff = torch.abs(emb_a - emb_b)
        combined = torch.cat([emb_a, emb_b, diff, endpoint_features], dim=1)
        return self.similarity_head(combined)

    def get_embeddings(self, seq_a, mask_a, seq_b, mask_b):
        """Get embeddings without computing similarity (for contrastive loss)."""
        return self.encoder(seq_a, mask_a), self.encoder(seq_b, mask_b)


if __name__ == "__main__":
    model = SiameseTransformerNetwork()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Transformer parameters: {n_params:,}")
    print(model)

    # Test forward pass
    seq_a = torch.randn(4, 20, 8)
    seq_b = torch.randn(4, 15, 8)
    mask_a = torch.zeros(4, 20, dtype=torch.bool)
    mask_b = torch.zeros(4, 15, dtype=torch.bool)
    mask_b[:, 12:] = True  # Simulate padding
    ep = torch.randn(4, 4)

    out = model(seq_a, mask_a, seq_b, mask_b, ep)
    print(f"Input: A={seq_a.shape}, B={seq_b.shape} -> Output: {out.shape}")
    print(f"Output: {out.squeeze().detach()}")
