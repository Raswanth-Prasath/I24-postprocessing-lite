"""
Physics-informed transformer cost model for stitching.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.physics_residuals import RESIDUAL_DIM
    from models.transformer_model import TransformerTrajectoryEncoder
except ImportError:
    from physics_residuals import RESIDUAL_DIM  # type: ignore
    from transformer_model import TransformerTrajectoryEncoder  # type: ignore


class PhysicsInformedCostNetwork(nn.Module):
    """
    Total cost:
        c = sum_i softplus(w_i(combined)) * residual_i + softplus(delta_theta(combined))

    `residual_i` are deterministic transformed physics residuals provided as input.
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
        residual_dim: int = RESIDUAL_DIM,
        use_correction: bool = True,
        pool_weight_first: float = 0.2,
        pool_weight_last: float = 0.5,
        pool_weight_mean: float = 0.3,
    ):
        super().__init__()

        self.encoder = TransformerTrajectoryEncoder(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
            pool_weight_first=pool_weight_first,
            pool_weight_last=pool_weight_last,
            pool_weight_mean=pool_weight_mean,
        )
        emb_size = self.encoder.output_size
        self.combined_dim = emb_size * 3 + endpoint_dim
        self.residual_dim = int(residual_dim)
        self.use_correction = bool(use_correction)

        self.physics_weight_head = nn.Linear(self.combined_dim, self.residual_dim)

        self.correction_head = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Training-only regularizer head.
        self.aux_head = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.residual_dim),
        )

        # Dedicated gate logit head; intentionally unconstrained.
        self.gate_head = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _encode_pair(
        self,
        seq_a: torch.Tensor,
        mask_a: Optional[torch.Tensor],
        seq_b: torch.Tensor,
        mask_b: Optional[torch.Tensor],
        endpoint_features: torch.Tensor,
    ) -> torch.Tensor:
        emb_a = self.encoder(seq_a, mask_a)
        emb_b = self.encoder(seq_b, mask_b)
        diff = torch.abs(emb_a - emb_b)
        return torch.cat([emb_a, emb_b, diff, endpoint_features], dim=1)

    def forward(
        self,
        seq_a: torch.Tensor,
        mask_a: Optional[torch.Tensor],
        seq_b: torch.Tensor,
        mask_b: Optional[torch.Tensor],
        endpoint_features: torch.Tensor,
        physics_residuals: torch.Tensor,
        return_gate_logits: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Args:
            physics_residuals: (batch, residual_dim), transformed deterministic residuals.

        Returns:
            total_cost: (batch, 1)
            aux_pred: (batch, residual_dim)
            weights: (batch, residual_dim), non-negative after softplus
            gate_logits: (batch, 1), only when return_gate_logits=True
        """
        combined = self._encode_pair(seq_a, mask_a, seq_b, mask_b, endpoint_features)
        weights = F.softplus(self.physics_weight_head(combined))
        physics_cost = torch.sum(weights * physics_residuals, dim=1, keepdim=True)

        if self.use_correction:
            delta = F.softplus(self.correction_head(combined))
        else:
            delta = torch.zeros_like(physics_cost)

        total_cost = physics_cost + delta
        aux_pred = self.aux_head(combined)
        if return_gate_logits:
            gate_logits = self.gate_head(combined)
            return total_cost, aux_pred, weights, gate_logits
        return total_cost, aux_pred, weights

    def inference_cost(
        self,
        seq_a: torch.Tensor,
        mask_a: Optional[torch.Tensor],
        seq_b: torch.Tensor,
        mask_b: Optional[torch.Tensor],
        endpoint_features: torch.Tensor,
        physics_residuals: torch.Tensor,
    ) -> torch.Tensor:
        total, _aux, _w = self.forward(
            seq_a=seq_a,
            mask_a=mask_a,
            seq_b=seq_b,
            mask_b=mask_b,
            endpoint_features=endpoint_features,
            physics_residuals=physics_residuals,
        )
        return total
