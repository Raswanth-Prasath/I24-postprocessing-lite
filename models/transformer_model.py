"""
Siamese Transformer for Trajectory Fragment Stitching

Architecture:
  Input: (batch, seq_len, 8) raw features
  -> Linear projection (8 -> 64)
  -> Sinusoidal positional encoding + learned time-aware bias (from t_norm)
  -> TransformerEncoder (2 layers, 4 heads, d_ff=128)
  -> Endpoint-weighted pooling (first/last/mean) -> 64-dim embedding
  -> Similarity head with |emb_a - emb_b| + endpoint features
     - classification objective: sigmoid probability
     - ranking objective: raw scalar score (lower = better)

~55K parameters
"""

import math
import torch
import torch.nn as nn
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


class TimeAwarePositionalBias(nn.Module):
    """
    Projects continuous time feature (t_norm) into embedding space.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._init_legacy_safe()

    def _init_legacy_safe(self):
        """
        Keep initial output near zero so checkpoints trained before this module
        (without time_bias weights) retain behavior after non-strict loading.
        """
        last = self.proj[2]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_norm: (batch, seq_len, 1)
        Returns:
            (batch, seq_len, d_model) additive bias
        """
        return self.proj(t_norm)


class TransformerTrajectoryEncoder(nn.Module):
    """
    Transformer encoder for variable-length trajectory sequences.

    Linear(8, 64) -> PositionalEncoding + TimeAwarePositionalBias
    -> TransformerEncoder -> endpoint-weighted pooling
    """

    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pool_weight_first: float = 0.2,
        pool_weight_last: float = 0.5,
        pool_weight_mean: float = 0.3,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_size = input_size
        self.time_feature_index = input_size - 1  # t_norm is final feature in rich sequence
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.time_bias = TimeAwarePositionalBias(d_model)

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
        # Endpoint-aware pooling emphasizes fragment boundaries critical for stitching.
        self.pool_weight_first = float(pool_weight_first)
        self.pool_weight_last = float(pool_weight_last)
        self.pool_weight_mean = float(pool_weight_mean)
        self._validate_pooling_weights()

    def _validate_pooling_weights(self) -> None:
        weights = (self.pool_weight_first, self.pool_weight_last, self.pool_weight_mean)
        if any(w < 0 for w in weights):
            raise ValueError(f"Pooling weights must be non-negative, got {weights}.")
        if not all(math.isfinite(w) for w in weights):
            raise ValueError(f"Pooling weights must be finite, got {weights}.")
        if sum(weights) <= 0:
            raise ValueError(f"At least one pooling weight must be > 0, got {weights}.")

    def _encode_input_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw input features into transformer token representations.
        """
        if x.size(-1) <= self.time_feature_index:
            raise ValueError(
                f"Expected at least {self.time_feature_index + 1} features per token, "
                f"got {x.size(-1)}."
            )

        # Extract time feature before projection so time bias is learned from raw t_norm.
        t_norm = x[:, :, self.time_feature_index:self.time_feature_index + 1]

        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = x + self.time_bias(t_norm)
        return x

    def _endpoint_weighted_pool(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Endpoint-weighted pooling over valid timesteps.
        """
        batch_size, seq_len, feat_dim = x.shape

        if padding_mask is not None:
            valid_mask = ~padding_mask
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()
            x_masked = x * valid_mask_expanded
            lengths = valid_mask.sum(dim=1).clamp(min=1)  # (batch,)
            mean_pool = x_masked.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            lengths = torch.full((batch_size,), seq_len, device=x.device, dtype=torch.long)
            mean_pool = x.mean(dim=1)

        # Endpoints are taken from the raw token stream. For right-padded sequences this
        # captures real boundary states, while mean pooling still ignores padded tokens.
        first_idx = torch.zeros(batch_size, 1, 1, device=x.device, dtype=torch.long).expand(-1, 1, feat_dim)
        first_tok = torch.gather(x, dim=1, index=first_idx).squeeze(1)

        # For length-1 sequences, last_pos becomes 0 so first_tok == last_tok.
        last_pos = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, feat_dim)
        last_tok = torch.gather(x, dim=1, index=last_pos).squeeze(1)

        w_first = self.pool_weight_first
        w_last = self.pool_weight_last
        w_mean = self.pool_weight_mean
        # Keep normalization robust even if custom configs are close to zero.
        w_sum = max(w_first + w_last + w_mean, 1e-8)

        pooled = (w_first * first_tok + w_last * last_tok + w_mean * mean_pool) / w_sum
        return pooled

    def _forward_layer_with_attention(
        self,
        layer: nn.TransformerEncoderLayer,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ):
        """
        Manual encoder layer forward used only for attention extraction.
        """
        if layer.norm_first:
            y = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(
                y,
                y,
                y,
                attn_mask=None,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + layer.dropout1(attn_out)

            y = layer.norm2(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))
            x = x + layer.dropout2(ff_out)
        else:
            attn_out, attn_weights = layer.self_attn(
                x,
                x,
                x,
                attn_mask=None,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = layer.norm1(x + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_out))

        return x, attn_weights

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            padding_mask: (batch, seq_len) True = padded positions to ignore
        Returns:
            (batch, d_model) embedding via endpoint-weighted pooling
        """
        x = self._encode_input_tokens(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        pooled = self._endpoint_weighted_pool(x, padding_mask)
        return self.layer_norm(pooled)

    def encode_with_attention(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        layer_index: int = -1,
        average_heads: bool = True,
    ):
        """
        Eval-oriented encoder path that also returns self-attention maps.

        Returns:
            embedding: (batch, d_model)
            attention: (batch, seq_len, seq_len) if averaged heads,
                       else (batch, heads, seq_len, seq_len)
        """
        x = self._encode_input_tokens(x)
        all_attn = []

        for layer in self.transformer.layers:
            x, attn = self._forward_layer_with_attention(layer, x, padding_mask)
            all_attn.append(attn)

        if self.transformer.norm is not None:
            x = self.transformer.norm(x)

        pooled = self._endpoint_weighted_pool(x, padding_mask)
        embedding = self.layer_norm(pooled)

        if not all_attn:
            return embedding, None

        if layer_index < 0:
            layer_index = len(all_attn) + layer_index
        layer_index = max(0, min(layer_index, len(all_attn) - 1))

        attn = all_attn[layer_index]
        if average_heads and attn is not None and attn.dim() == 4:
            attn = attn.mean(dim=1)

        return embedding, attn


class SiameseTransformerNetwork(nn.Module):
    """
    Siamese Transformer for trajectory fragment similarity.

    Twin encoders (shared) -> [emb_a; emb_b; |emb_a - emb_b|; endpoint_feats]
    -> similarity head -> probability (classification) or raw score (ranking)

    Key difference from TCN: adds |emb_a - emb_b| element-wise difference.
    """

    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        endpoint_dim: int = 4,
        training_objective: str = "classification",
        pool_weight_first: float = 0.2,
        pool_weight_last: float = 0.5,
        pool_weight_mean: float = 0.3,
    ):
        super().__init__()

        self.encoder = TransformerTrajectoryEncoder(
            input_size=input_size, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=num_layers, dropout=dropout,
            pool_weight_first=pool_weight_first,
            pool_weight_last=pool_weight_last,
            pool_weight_mean=pool_weight_mean,
        )

        emb_size = self.encoder.output_size  # 64
        # [emb_a; emb_b; |emb_a - emb_b|; endpoint_feats] = 64 + 64 + 64 + 4 = 196
        combined_dim = emb_size * 3 + endpoint_dim

        objective = str(training_objective or "classification").lower()
        if objective not in {"classification", "ranking"}:
            raise ValueError(
                f"training_objective must be 'classification' or 'ranking', got '{training_objective}'."
            )
        self.training_objective = objective

        head_layers = [
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        ]
        if self.training_objective == "classification":
            head_layers.append(nn.Sigmoid())
        self.similarity_head = nn.Sequential(*head_layers)

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
            (batch, 1) similarity probability for classification objective,
            or raw score for ranking objective (lower score = better match).
        """
        emb_a = self.encoder(seq_a, mask_a)
        emb_b = self.encoder(seq_b, mask_b)
        diff = torch.abs(emb_a - emb_b)
        combined = torch.cat([emb_a, emb_b, diff, endpoint_features], dim=1)
        return self.similarity_head(combined)

    def get_embeddings(self, seq_a, mask_a, seq_b, mask_b):
        """Get embeddings without computing similarity (for contrastive loss)."""
        return self.encoder(seq_a, mask_a), self.encoder(seq_b, mask_b)

    def get_attention_maps(self, seq_a, mask_a, seq_b, mask_b,
                           layer_index: int = -1, average_heads: bool = True):
        """
        Return attention maps for both branches.
        """
        _, attn_a = self.encoder.encode_with_attention(
            seq_a, mask_a, layer_index=layer_index, average_heads=average_heads
        )
        _, attn_b = self.encoder.encode_with_attention(
            seq_b, mask_b, layer_index=layer_index, average_heads=average_heads
        )
        return attn_a, attn_b


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

    _, attn_a = model.encoder.encode_with_attention(seq_a, mask_a)
    if attn_a is not None:
        print(f"Attention map shape (A): {attn_a.shape}")
