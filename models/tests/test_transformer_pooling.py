import pytest
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_model import SiameseTransformerNetwork, TransformerTrajectoryEncoder


def _build_encoder(**kwargs):
    return TransformerTrajectoryEncoder(
        input_size=8,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        num_layers=1,
        dropout=0.0,
        **kwargs,
    )


def test_endpoint_weighted_pool_no_padding():
    encoder = _build_encoder(pool_weight_first=0.2, pool_weight_last=0.5, pool_weight_mean=0.3)
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]])

    pooled = encoder._endpoint_weighted_pool(x, padding_mask=None)
    expected = torch.tensor([[3.6, 3.6]])  # 0.2*first + 0.5*last + 0.3*mean
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_endpoint_weighted_pool_len1_with_padding():
    encoder = _build_encoder(pool_weight_first=0.2, pool_weight_last=0.5, pool_weight_mean=0.3)
    x = torch.tensor([[[2.0, 4.0], [100.0, 100.0], [200.0, 200.0]]])
    mask = torch.tensor([[False, True, True]])

    pooled = encoder._endpoint_weighted_pool(x, padding_mask=mask)
    expected = torch.tensor([[2.0, 4.0]])
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_pooling_weights_configurable_through_model_ctor():
    model = SiameseTransformerNetwork(
        pool_weight_first=0.1,
        pool_weight_last=0.6,
        pool_weight_mean=0.3,
    )
    assert model.encoder.pool_weight_first == pytest.approx(0.1)
    assert model.encoder.pool_weight_last == pytest.approx(0.6)
    assert model.encoder.pool_weight_mean == pytest.approx(0.3)


def test_invalid_pooling_weights_raise():
    with pytest.raises(ValueError):
        _build_encoder(pool_weight_first=0.0, pool_weight_last=0.0, pool_weight_mean=0.0)

    with pytest.raises(ValueError):
        _build_encoder(pool_weight_first=-0.1, pool_weight_last=0.5, pool_weight_mean=0.6)
