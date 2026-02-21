from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.heteroscedastic_model import HeteroscedasticTrajectoryModel


def _build_model() -> HeteroscedasticTrajectoryModel:
    return HeteroscedasticTrajectoryModel(
        input_size=8,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        dropout=0.0,
        pred_hidden=64,
        cov_eps=1e-3,
    )


def test_forward_shapes_with_padding():
    torch.manual_seed(0)
    model = _build_model()

    seq = torch.randn(3, 7, 8)
    mask = torch.tensor(
        [
            [False, False, False, False, False, True, True],
            [False, False, False, False, False, False, False],
            [False, False, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    query_dt = torch.tensor(
        [
            [0.1, 0.2, 0.4, 0.8],
            [0.2, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    mu, chol = model(seq, mask, query_dt)
    assert mu.shape == (3, 4, 2)
    assert chol.shape == (3, 4, 3)


def test_cholesky_builds_positive_definite_covariance():
    torch.manual_seed(1)
    model = _build_model()
    raw = torch.randn(5, 6, 3)

    L = model.build_cholesky(raw)
    cov = torch.matmul(L, L.transpose(-1, -2))

    det = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]
    assert torch.all(det > 0)
    assert torch.all(torch.isfinite(det))


def test_nll_and_mahalanobis_finite_with_query_mask():
    torch.manual_seed(2)
    model = _build_model()

    seq = torch.randn(2, 6, 8)
    seq_mask = torch.zeros(2, 6, dtype=torch.bool)
    qdt = torch.tensor([[0.1, 0.2, 0.4], [0.1, 0.3, 0.0]], dtype=torch.float32)
    qmask = torch.tensor([[False, False, False], [False, False, True]], dtype=torch.bool)

    mu, chol = model(seq, seq_mask, qdt)
    targets = mu + 0.05 * torch.randn_like(mu)

    nll = model.compute_nll(mu, chol, targets, qmask)
    maha = model.compute_mahalanobis(mu, chol, targets, qmask)

    assert nll.ndim == 0
    assert torch.isfinite(nll)
    assert maha.shape == (2,)
    assert torch.all(torch.isfinite(maha))


def test_mahalanobis_is_lower_for_exact_targets_than_perturbed():
    torch.manual_seed(3)
    model = _build_model()

    seq = torch.randn(1, 5, 8)
    seq_mask = torch.zeros(1, 5, dtype=torch.bool)
    qdt = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    qmask = torch.zeros(1, 3, dtype=torch.bool)

    mu, chol = model(seq, seq_mask, qdt)

    exact = model.compute_mahalanobis(mu, chol, mu, qmask)
    perturbed = model.compute_mahalanobis(mu, chol, mu + 0.5, qmask)

    assert exact.item() <= perturbed.item()


def test_single_optimizer_step_backpropagates():
    torch.manual_seed(4)
    model = _build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    seq = torch.randn(4, 8, 8)
    seq_mask = torch.zeros(4, 8, dtype=torch.bool)
    qdt = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.3, 0.0, 0.0],
            [0.1, 0.2, 0.4, 0.8],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor(
        [
            [False, False, False, True],
            [False, False, False, False],
            [False, False, True, True],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )

    mu, chol = model(seq, seq_mask, qdt)
    targets = mu.detach() + 0.1 * torch.randn_like(mu)

    loss = model.compute_nll(mu, chol, targets, qmask)
    assert torch.isfinite(loss)

    opt.zero_grad()
    loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().abs().sum().item())
    assert grad_norm > 0.0

    opt.step()
