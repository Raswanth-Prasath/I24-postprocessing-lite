"""
Heteroscedastic transformer model for trajectory stitching.

Model predicts relative future offsets and full 2x2 covariance for each query time.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.transformer_model import TransformerTrajectoryEncoder
except ImportError:
    from transformer_model import TransformerTrajectoryEncoder  # type: ignore


class HeteroscedasticTrajectoryModel(nn.Module):
    """
    Encode one trajectory, then predict relative position + covariance at query times.

    Outputs are in relative coordinates (offset from track1 endpoint):
      mu = [dx, dy]
      covariance = L @ L.T where L is lower-triangular from learned parameters
    """

    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pred_hidden: int = 128,
        cov_eps: float = 1e-3,
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

        self.predict_head = nn.Sequential(
            nn.Linear(emb_size + 1, pred_hidden),
            nn.GELU(),
            nn.Linear(pred_hidden, pred_hidden // 2),
            nn.GELU(),
            nn.Linear(pred_hidden // 2, 5),  # [mu_x, mu_y, a, b, c]
        )

        self.cov_eps = float(cov_eps)
        if self.cov_eps <= 0:
            raise ValueError(f"cov_eps must be > 0, got {self.cov_eps}")

    def encode(self, seq: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode input sequence into one embedding per sample."""
        return self.encoder(seq, mask)

    def predict_from_embedding(
        self,
        embedding: torch.Tensor,
        query_dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean offsets and covariance params from cached embeddings.

        Args:
            embedding: (B, E)
            query_dt: (B, Q)
        Returns:
            mu: (B, Q, 2)
            chol_params: (B, Q, 3) for lower-triangular [l11_raw, l21, l22_raw]
        """
        if query_dt.dim() != 2:
            raise ValueError(f"query_dt must have shape (B, Q), got {tuple(query_dt.shape)}")
        if embedding.size(0) != query_dt.size(0):
            raise ValueError(
                f"embedding batch ({embedding.size(0)}) != query_dt batch ({query_dt.size(0)})"
            )

        bsz, n_query = query_dt.shape
        emb_expanded = embedding.unsqueeze(1).expand(bsz, n_query, -1)
        dt_expanded = query_dt.unsqueeze(-1)
        head_inp = torch.cat([emb_expanded, dt_expanded], dim=-1)
        pred = self.predict_head(head_inp)

        mu = pred[..., :2]
        chol_params = pred[..., 2:]
        return mu, chol_params

    def forward(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor],
        query_dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(seq, mask)
        return self.predict_from_embedding(embedding, query_dt)

    def build_cholesky(self, chol_params: torch.Tensor) -> torch.Tensor:
        """
        Convert raw Cholesky parameters to lower-triangular matrix L.

        chol_params[..., 0] -> l11_raw, positive after softplus + cov_eps
        chol_params[..., 1] -> l21
        chol_params[..., 2] -> l22_raw, positive after softplus + cov_eps
        """
        l11 = F.softplus(chol_params[..., 0]) + self.cov_eps
        l21 = chol_params[..., 1]
        l22 = F.softplus(chol_params[..., 2]) + self.cov_eps

        l11 = l11.clamp(max=1e3)
        l22 = l22.clamp(max=1e3)

        L = torch.zeros(*chol_params.shape[:-1], 2, 2, device=chol_params.device, dtype=chol_params.dtype)
        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 1, 1] = l22
        return L

    @staticmethod
    def _masked_mean(values: torch.Tensor, query_mask: Optional[torch.Tensor], dim: Tuple[int, ...]) -> torch.Tensor:
        """
        Compute masked mean over requested dims.

        query_mask convention: True means padded/invalid.
        """
        if query_mask is None:
            for d in sorted(dim, reverse=True):
                values = values.mean(dim=d)
            return values

        valid = (~query_mask).to(values.dtype)
        while valid.dim() < values.dim():
            valid = valid.unsqueeze(-1)
        masked = values * valid

        reduce_mask = valid
        denom = reduce_mask
        for d in sorted(dim, reverse=True):
            masked = masked.sum(dim=d)
            denom = denom.sum(dim=d)
        return masked / denom.clamp_min(1.0)

    def per_point_mahalanobis(
        self,
        mu: torch.Tensor,
        chol_params: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-point Mahalanobis distance with shape (B, Q)."""
        if mu.shape != targets.shape:
            raise ValueError(f"mu shape {tuple(mu.shape)} must match targets shape {tuple(targets.shape)}")

        L = self.build_cholesky(chol_params)
        err = targets - mu

        z = torch.linalg.solve_triangular(
            L,
            err.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        return (z * z).sum(dim=-1)

    def compute_mahalanobis(
        self,
        mu: torch.Tensor,
        chol_params: torch.Tensor,
        targets: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mean Mahalanobis score per sample.

        Returns:
            Tensor with shape (B,)
        """
        maha = self.per_point_mahalanobis(mu, chol_params, targets)
        return self._masked_mean(maha, query_mask=query_mask, dim=(1,))

    def compute_nll(
        self,
        mu: torch.Tensor,
        chol_params: torch.Tensor,
        targets: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gaussian NLL under full covariance.

        Returns:
            Scalar mean NLL over all valid points in batch.
        """
        L = self.build_cholesky(chol_params)
        maha = self.per_point_mahalanobis(mu, chol_params, targets)

        logdet = 2.0 * (torch.log(L[..., 0, 0]) + torch.log(L[..., 1, 1]))
        nll_per_point = 0.5 * (maha + logdet + 2.0 * math.log(2.0 * math.pi))

        per_sample = self._masked_mean(nll_per_point, query_mask=query_mask, dim=(1,))
        return per_sample.mean()
