import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.evaluate_transformer import (  # noqa: E402
    LoadedModel,
    _resolve_ranking_dataset_path,
    _resolve_ranking_output_path,
)


def _dummy_loaded(train_config):
    return LoadedModel(
        model=torch.nn.Identity(),
        seq_mean=torch.zeros(8),
        seq_std=torch.ones(8),
        ep_mean=torch.zeros(4),
        ep_std=torch.ones(4),
        model_config={},
        train_config=train_config,
        checkpoint_path=Path("dummy_checkpoint.pth"),
    )


def test_resolve_ranking_dataset_path_prefers_cli_override(tmp_path):
    dataset_path = tmp_path / "ranking.jsonl"
    args = SimpleNamespace(dataset_path=str(dataset_path))
    loaded = _dummy_loaded({"dataset_path": "ignored.jsonl"})
    assert _resolve_ranking_dataset_path(loaded, args) == dataset_path


def test_resolve_ranking_dataset_path_uses_checkpoint_train_config_when_cli_missing():
    args = SimpleNamespace(dataset_path="")
    loaded = _dummy_loaded({"dataset_path": "models/outputs/from_checkpoint.jsonl"})
    assert _resolve_ranking_dataset_path(loaded, args) == PROJECT_ROOT / "models/outputs/from_checkpoint.jsonl"


def test_resolve_ranking_dataset_path_falls_back_to_default_when_missing():
    args = SimpleNamespace(dataset_path="")
    loaded = _dummy_loaded({})
    assert _resolve_ranking_dataset_path(loaded, args) == PROJECT_ROOT / "models/outputs/transformer_ranking_dataset.jsonl"


def test_resolve_ranking_output_path_defaults_to_split_mode():
    args = SimpleNamespace(ranking_output="", replay=False, ranking_split="val", ranking_scenario="i")
    assert _resolve_ranking_output_path(args) == PROJECT_ROOT / "models/outputs/transformer_ranking_gate_val.json"


def test_resolve_ranking_output_path_defaults_to_replay_scenario():
    args = SimpleNamespace(ranking_output="", replay=True, ranking_split="val", ranking_scenario="iii")
    assert _resolve_ranking_output_path(args) == PROJECT_ROOT / "models/outputs/transformer_ranking_gate_iii.json"


def test_resolve_ranking_output_path_uses_explicit_override(tmp_path):
    out_path = tmp_path / "custom_gate.json"
    args = SimpleNamespace(ranking_output=str(out_path), replay=False, ranking_split="val", ranking_scenario="i")
    assert _resolve_ranking_output_path(args) == out_path

