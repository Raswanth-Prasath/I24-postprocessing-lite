"""
Evaluation and Calibration Script for Siamese Transformer Model

This script provides:
1. Standard classification evaluation (ROC-AUC, AP, accuracy, confusion matrix)
2. Attention map export for explainability
3. Pipeline-based cost calibration against Bhattacharyya cost distribution
4. Complementarity audit against Bhattacharyya on pipeline candidate pairs
"""

import argparse
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rich_sequence_dataset import (  # noqa: E402
    RichSequenceDataset,
    _get_gt_id,
    extract_endpoint_features,
    extract_rich_sequence,
    rich_collate_fn,
)
from models.transformer_model import SiameseTransformerNetwork  # noqa: E402


@dataclass
class LoadedModel:
    model: SiameseTransformerNetwork
    seq_mean: torch.Tensor
    seq_std: torch.Tensor
    ep_mean: torch.Tensor
    ep_std: torch.Tensor
    model_config: Dict
    train_config: Dict
    checkpoint_path: Path


class RecordingCostFunction:
    """
    Decorator cost function used to record every candidate pair evaluated
    by MOTGraphSingle during stitching.
    """

    def __init__(self, delegate):
        self.delegate = delegate
        self.records: List[Tuple[dict, dict, float, float]] = []

    def compute_cost(self, track1: dict, track2: dict, TIME_WIN: float, param: dict) -> float:
        cost = float(self.delegate.compute_cost(track1, track2, TIME_WIN, param))
        gap = float(track2["timestamp"][0] - track1["timestamp"][-1])
        self.records.append((track1, track2, cost, gap))
        return cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and calibrate Siamese Transformer model.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_stitch_model.pth"),
        help="Path to trained transformer checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=["iii"],
        help="Dataset scenarios for classification evaluation (e.g., iii or i ii iii).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for evaluation.",
    )
    parser.add_argument(
        "--export-attention",
        action="store_true",
        help="Export attention heatmaps from selected evaluation pairs.",
    )
    parser.add_argument(
        "--max-attention-pairs",
        type=int,
        default=16,
        help="Maximum number of pairs to export attention maps for.",
    )
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help="Fit a calibration mapping from transformer cost to Bhattacharyya-like cost.",
    )
    parser.add_argument(
        "--calibration-scenario",
        type=str,
        default="i",
        help="Scenario used for pipeline pair collection during calibration fitting.",
    )
    parser.add_argument(
        "--calibration-mode",
        type=str,
        default="isotonic",
        choices=["linear", "isotonic", "quantile_match"],
        help="Calibration mapping mode.",
    )
    parser.add_argument(
        "--calibration-output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_calibration.json"),
        help="Path to write calibration JSON artifact.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=5.0,
        help="Scale factor used in raw transformer base cost after similarity mapping.",
    )
    parser.add_argument(
        "--time-penalty",
        type=float,
        default=0.1,
        help="Time penalty used in runtime total cost calculation.",
    )
    parser.add_argument(
        "--similarity-mapping",
        type=str,
        default="linear",
        choices=["linear", "odds", "power"],
        help="Mapping from similarity score to dissimilarity before scaling.",
    )
    parser.add_argument(
        "--similarity-power",
        type=float,
        default=1.0,
        help="Power exponent for --similarity-mapping=power.",
    )
    parser.add_argument(
        "--similarity-clip-eps",
        type=float,
        default=1e-2,
        help="Clipping epsilon applied to similarity before mapping.",
    )
    parser.add_argument(
        "--training-objective",
        type=str,
        default="classification",
        choices=["classification", "ranking"],
        help="Expected checkpoint objective for score-to-cost conversion.",
    )
    parser.add_argument(
        "--score-mapping",
        type=str,
        default="auto",
        choices=["auto", "legacy_similarity", "direct_cost"],
        help="Mapping from model output to raw base cost.",
    )
    parser.add_argument(
        "--max-calibration-pairs",
        type=int,
        default=0,
        help="Optional cap on calibration pairs (0 = use all collected pairs).",
    )
    parser.add_argument(
        "--calibration-target-max",
        type=float,
        default=10.0,
        help=(
            "Upper bound for Bhattacharyya base cost used during calibration fitting. "
            "Use <=0 to disable explicit cap."
        ),
    )
    parser.add_argument(
        "--audit-complementarity",
        action="store_true",
        help="Run complementarity audit against Bhattacharyya on pipeline candidate pairs.",
    )
    parser.add_argument(
        "--audit-scenario",
        type=str,
        default="i",
        help="Scenario to replay for complementarity audit.",
    )
    parser.add_argument(
        "--audit-output",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_complementarity_i.json"),
        help="Path to write complementarity audit JSON artifact.",
    )
    parser.add_argument(
        "--audit-plots-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs"),
        help="Directory for complementarity audit plots.",
    )
    parser.add_argument(
        "--audit-min-candidates",
        type=int,
        default=2,
        help="Minimum candidates per anchor required for anchor-level ranking metrics.",
    )
    parser.add_argument(
        "--audit-go-lift-threshold",
        type=float,
        default=0.15,
        help="Minimum Bhattacharyya-failure recovery rate required to recommend continuing transformer investment.",
    )
    parser.add_argument(
        "--audit-max-pairs",
        type=int,
        default=0,
        help="Optional cap on complementarity audit pairs (0 = use all collected pairs).",
    )
    parser.add_argument(
        "--evaluate-ranking-gates",
        action="store_true",
        help="Evaluate ranking gates (pairwise/top1/spearman) on ranking dataset split rows.",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Use pipeline replay pairs instead of dataset split rows for ranking gates.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Optional ranking dataset JSONL path override for ranking-gate evaluation.",
    )
    parser.add_argument(
        "--ranking-split",
        type=str,
        default="val",
        choices=["train", "val", "test", "all"],
        help="Dataset split used for ranking-gate evaluation when --replay is not set.",
    )
    parser.add_argument(
        "--ranking-scenario",
        type=str,
        default="i",
        help="Scenario to replay when --replay is set.",
    )
    parser.add_argument(
        "--ranking-output",
        type=str,
        default="",
        help="Path to write ranking gate evaluation artifact. Defaults depend on source mode.",
    )
    parser.add_argument(
        "--ranking-max-pairs",
        type=int,
        default=0,
        help="Optional cap on ranking candidate rows/pairs (0 = use all available).",
    )
    parser.add_argument(
        "--ranking-min-candidates",
        type=int,
        default=2,
        help="Minimum candidates per anchor for ranking gate metrics.",
    )
    parser.add_argument(
        "--gate-pairwise-threshold",
        type=float,
        default=0.95,
        help="Gate threshold for GT cross-class pairwise accuracy.",
    )
    parser.add_argument(
        "--gate-top1-threshold",
        type=float,
        default=0.85,
        help="Gate threshold for anchor Top-1 accuracy.",
    )
    parser.add_argument(
        "--gate-spearman-threshold",
        type=float,
        default=0.55,
        help="Gate threshold for mean per-anchor Spearman correlation.",
    )
    return parser.parse_args()


def _resolve_raw_path(scenario: str) -> Path:
    candidates = [
        PROJECT_ROOT / f"RAW_{scenario}.json",
        PROJECT_ROOT / "Siamese-Network" / "data" / f"RAW_{scenario}.json",
        PROJECT_ROOT / "Siamese-Network" / "data" / f"RAW_{scenario}_Bhat.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find RAW data for scenario '{scenario}'. Checked: {candidates}")


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _load_utils_modules_for_motgraph():
    """
    Load utils modules by file path to avoid importing utils/__init__.py,
    which depends on optional packages not required for calibration replay.
    """
    if "i24_logger" not in sys.modules:
        logger_pkg = types.ModuleType("i24_logger")
        logger_pkg.__path__ = []
        sys.modules["i24_logger"] = logger_pkg

    if "i24_logger.log_writer" not in sys.modules:
        log_writer = types.ModuleType("i24_logger.log_writer")

        def catch_critical(errors=(Exception,)):
            def decorator(func):
                return func
            return decorator

        class _NoopLogger:
            def set_name(self, *args, **kwargs):
                return None

            def __getattr__(self, _name):
                def _noop(*args, **kwargs):
                    return None
                return _noop

        log_writer.catch_critical = catch_critical
        log_writer.logger = _NoopLogger()
        sys.modules["i24_logger.log_writer"] = log_writer

    utils_dir = PROJECT_ROOT / "utils"
    if "utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(utils_dir)]
        sys.modules["utils"] = utils_pkg

    modules = {}
    for name in ["utils_stitcher_cost", "stitch_cost_interface", "utils_mcf"]:
        full_name = f"utils.{name}"
        if full_name in sys.modules:
            modules[name] = sys.modules[full_name]
            continue
        modules[name] = _load_module_from_path(full_name, utils_dir / f"{name}.py")

    return modules["utils_mcf"], modules["stitch_cost_interface"]


def load_model(checkpoint_path: Path, device: torch.device) -> LoadedModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    train_config = checkpoint.get("train_config", {})
    if not isinstance(train_config, dict):
        train_config = {}
    model = SiameseTransformerNetwork(**model_config)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    legacy_time_bias_keys = {
        "encoder.time_bias.proj.0.weight",
        "encoder.time_bias.proj.0.bias",
        "encoder.time_bias.proj.2.weight",
        "encoder.time_bias.proj.2.bias",
    }
    missing_set = set(load_result.missing_keys)
    unexpected_set = set(load_result.unexpected_keys)
    if missing_set and missing_set.issubset(legacy_time_bias_keys) and not unexpected_set:
        print("Info: legacy checkpoint detected (no time_bias weights); using safe zero-initialized time bias.")
    else:
        if load_result.missing_keys:
            print(f"Warning: missing keys when loading checkpoint: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: unexpected keys when loading checkpoint: {load_result.unexpected_keys}")
    model = model.to(device)
    model.eval()

    seq_mean = torch.as_tensor(
        checkpoint.get("seq_mean", np.zeros(8)),
        device=device,
        dtype=torch.float32,
    )
    seq_std = torch.as_tensor(
        checkpoint.get("seq_std", np.ones(8)),
        device=device,
        dtype=torch.float32,
    ).clamp_min(1e-6)
    ep_mean = torch.as_tensor(
        checkpoint.get("ep_mean", np.zeros(4)),
        device=device,
        dtype=torch.float32,
    )
    ep_std = torch.as_tensor(
        checkpoint.get("ep_std", np.ones(4)),
        device=device,
        dtype=torch.float32,
    ).clamp_min(1e-6)

    return LoadedModel(
        model=model,
        seq_mean=seq_mean,
        seq_std=seq_std,
        ep_mean=ep_mean,
        ep_std=ep_std,
        model_config=model_config,
        train_config=train_config,
        checkpoint_path=checkpoint_path,
    )


@torch.no_grad()
def evaluate_classification(model: SiameseTransformerNetwork,
                            loader: DataLoader,
                            device: torch.device,
                            seq_mean: torch.Tensor,
                            seq_std: torch.Tensor,
                            ep_mean: torch.Tensor,
                            ep_std: torch.Tensor) -> Dict:
    preds: List[float] = []
    labels: List[float] = []

    for pa, _la, ma, pb, _lb, mb, eps, y in loader:
        pa = pa.to(device)
        ma = ma.to(device)
        pb = pb.to(device)
        mb = mb.to(device)
        eps = eps.to(device)
        pa = (pa - seq_mean) / seq_std
        pb = (pb - seq_mean) / seq_std
        eps = (eps - ep_mean) / ep_std

        raw_output = model(pa, ma, pb, mb, eps).squeeze(1)
        if getattr(model, "training_objective", "classification") == "ranking":
            # Lower ranking score means better match; map to pseudo-probability for reporting.
            prob = torch.sigmoid(-raw_output)
        else:
            prob = raw_output
        preds.extend(prob.cpu().numpy().tolist())
        labels.extend(y.squeeze(1).cpu().numpy().tolist())

    preds_np = np.asarray(preds, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    pred_labels = (preds_np > 0.5).astype(np.int64)

    metrics = {
        "predictions": preds_np,
        "labels": labels_np,
        "pred_labels": pred_labels,
        "roc_auc": float(roc_auc_score(labels_np, preds_np)),
        "avg_precision": float(average_precision_score(labels_np, preds_np)),
        "accuracy": float(accuracy_score(labels_np, pred_labels)),
        "confusion_matrix": confusion_matrix(labels_np, pred_labels).tolist(),
        "classification_report": classification_report(
            labels_np,
            pred_labels,
            target_names=["Different Vehicle", "Same Vehicle"],
            zero_division=0,
            output_dict=True,
        ),
    }
    return metrics


def plot_standard_metrics(metrics: Dict, output_dir: Path):
    labels = metrics["labels"]
    preds = metrics["predictions"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    fpr, tpr, _ = roc_curve(labels, preds)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f"AUC={metrics['roc_auc']:.4f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    precision, recall, _ = precision_recall_curve(labels, preds)
    axes[0, 1].plot(recall, precision, linewidth=2, label=f"AP={metrics['avg_precision']:.4f}")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = axes[1, 0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title("Confusion Matrix (Normalized)")
    axes[1, 0].set_xticks([0, 1], labels=["Different", "Same"])
    axes[1, 0].set_yticks([0, 1], labels=["Different", "Same"])
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f"{int(cm[i, j])}\n({cm_norm[i, j]:.2%})",
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes[1, 0])

    labels_np = np.asarray(labels)
    pos = labels_np == 1
    neg = labels_np == 0
    axes[1, 1].hist(np.asarray(preds)[pos], bins=50, alpha=0.6, density=True, label="Same Vehicle")
    axes[1, 1].hist(np.asarray(preds)[neg], bins=50, alpha=0.6, density=True, label="Different Vehicle")
    axes[1, 1].axvline(0.5, linestyle="--", linewidth=2, color="black", label="Threshold 0.5")
    axes[1, 1].set_title("Similarity Distribution")
    axes[1, 1].set_xlabel("Similarity")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    out = output_dir / "transformer_evaluation.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def export_attention_maps(model: SiameseTransformerNetwork,
                          loader: DataLoader,
                          device: torch.device,
                          seq_mean: torch.Tensor,
                          seq_std: torch.Tensor,
                          ep_mean: torch.Tensor,
                          ep_std: torch.Tensor,
                          output_dir: Path,
                          max_pairs: int = 16):
    attn_dir = output_dir / "attention_maps"
    attn_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    metadata = []

    for pa, la, ma, pb, lb, mb, eps, y in loader:
        pa = pa.to(device)
        ma = ma.to(device)
        pb = pb.to(device)
        mb = mb.to(device)
        eps = eps.to(device)
        pa = (pa - seq_mean) / seq_std
        pb = (pb - seq_mean) / seq_std
        eps = (eps - ep_mean) / ep_std
        prob = model(pa, ma, pb, mb, eps).squeeze(1).cpu().numpy()

        batch_size = pa.size(0)
        for i in range(batch_size):
            if exported >= max_pairs:
                break

            len_a = int(la[i].item())
            len_b = int(lb[i].item())

            seq_a = pa[i:i + 1, :len_a]
            seq_b = pb[i:i + 1, :len_b]
            mask_a = torch.zeros(1, len_a, dtype=torch.bool, device=device)
            mask_b = torch.zeros(1, len_b, dtype=torch.bool, device=device)

            attn_a, attn_b = model.get_attention_maps(seq_a, mask_a, seq_b, mask_b)
            if attn_a is None or attn_b is None:
                continue

            attn_a_np = attn_a.squeeze(0).detach().cpu().numpy()[:len_a, :len_a]
            attn_b_np = attn_b.squeeze(0).detach().cpu().numpy()[:len_b, :len_b]

            fig_a, ax_a = plt.subplots(figsize=(6, 5))
            im_a = ax_a.imshow(attn_a_np, cmap="viridis", vmin=0, vmax=1)
            ax_a.set_title(f"Pair {exported} - Fragment A Attention")
            ax_a.set_xlabel("Key timestep")
            ax_a.set_ylabel("Query timestep")
            fig_a.colorbar(im_a, ax=ax_a)
            fig_a.tight_layout()
            fig_a.savefig(attn_dir / f"attention_pair_{exported}_A.png", dpi=220, bbox_inches="tight")
            plt.close(fig_a)

            fig_b, ax_b = plt.subplots(figsize=(6, 5))
            im_b = ax_b.imshow(attn_b_np, cmap="viridis", vmin=0, vmax=1)
            ax_b.set_title(f"Pair {exported} - Fragment B Attention")
            ax_b.set_xlabel("Key timestep")
            ax_b.set_ylabel("Query timestep")
            fig_b.colorbar(im_b, ax=ax_b)
            fig_b.tight_layout()
            fig_b.savefig(attn_dir / f"attention_pair_{exported}_B.png", dpi=220, bbox_inches="tight")
            plt.close(fig_b)

            metadata.append({
                "pair_index": exported,
                "prediction": float(prob[i]),
                "label": float(y[i].item()),
                "len_a": len_a,
                "len_b": len_b,
                "file_a": f"attention_pair_{exported}_A.png",
                "file_b": f"attention_pair_{exported}_B.png",
            })
            exported += 1

        if exported >= max_pairs:
            break

    with open(attn_dir / "attention_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {exported} attention map pairs to {attn_dir}")


def _rank_simple(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra = _rank_simple(a)
    rb = _rank_simple(b)
    corr = np.corrcoef(ra, rb)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def fit_calibration(raw_base: np.ndarray, target_base: np.ndarray, mode: str) -> Dict:
    raw_base = raw_base.astype(np.float64)
    target_base = target_base.astype(np.float64)

    if mode == "linear":
        x_knots = np.array([raw_base.min(), raw_base.max()], dtype=np.float64)
        y_knots = x_knots.copy()
    elif mode == "quantile_match":
        q = np.linspace(0, 1, 101)
        x_all = np.quantile(raw_base, q)
        y_all = np.quantile(target_base, q)
        x_knots, unique_idx = np.unique(x_all, return_index=True)
        y_knots = y_all[unique_idx]
    else:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_base, target_base)
        x_knots = np.asarray(iso.X_thresholds_, dtype=np.float64)
        y_knots = np.asarray(iso.y_thresholds_, dtype=np.float64)

    if len(x_knots) < 2:
        x_knots = np.array([raw_base.min(), raw_base.max()], dtype=np.float64)
        y_knots = np.array([target_base.min(), target_base.max()], dtype=np.float64)

    x_sorted = np.asarray(x_knots, dtype=np.float64)
    y_sorted = np.asarray(y_knots, dtype=np.float64)
    order = np.argsort(x_sorted)
    x_sorted = x_sorted[order]
    y_sorted = y_sorted[order]

    y_fit = np.interp(raw_base, x_sorted, y_sorted)
    mae = float(np.mean(np.abs(y_fit - target_base)))
    spearman = _spearman_corr(y_fit, target_base)

    return {
        "mode": mode,
        "x_knots": x_sorted.tolist(),
        "y_knots": y_sorted.tolist(),
        "fit_metrics": {
            "mae": mae,
            "spearman": spearman,
        },
    }


def apply_calibration(raw_base: np.ndarray, artifact: Dict) -> np.ndarray:
    x_knots = np.asarray(artifact["x_knots"], dtype=np.float64)
    y_knots = np.asarray(artifact["y_knots"], dtype=np.float64)
    domain = artifact.get("domain", [float(x_knots.min()), float(x_knots.max())])
    clipped = np.clip(raw_base, domain[0], domain[1])
    return np.interp(clipped, x_knots, y_knots)


def collect_pipeline_pairs_for_calibration(scenario: str) -> List[Tuple[dict, dict, float, float]]:
    """
    Collect realistic candidate pairs by replaying MOTGraphSingle and recording
    every Bhattacharyya cost evaluation.
    """
    utils_mcf_module, stitch_interface_module = _load_utils_modules_for_motgraph()
    MOTGraphSingle = utils_mcf_module.MOTGraphSingle
    BhattacharyyaCostFunction = stitch_interface_module.BhattacharyyaCostFunction

    raw_path = _resolve_raw_path(scenario)
    with open(raw_path, "r") as f:
        raw_fragments = json.load(f)

    fragments = []
    for idx, frag in enumerate(raw_fragments):
        frag_copy = dict(frag)
        if "local_fragment_id" in frag_copy:
            calib_id = f"local_{frag_copy['local_fragment_id']}"
        else:
            oid = frag_copy.get("_id")
            if isinstance(oid, dict) and "$oid" in oid:
                calib_id = oid["$oid"]
            else:
                calib_id = str(oid) if oid is not None else f"frag_{idx}"
        frag_copy["calib_id"] = calib_id
        frag_copy["compute_node_id"] = str(frag_copy.get("compute_node_id", "0"))
        for key in ("timestamp", "x_position", "y_position", "velocity"):
            if key in frag_copy and frag_copy[key] is not None:
                frag_copy[key] = np.asarray(frag_copy[key], dtype=np.float64)
        fragments.append(frag_copy)

    by_direction: Dict[str, List[dict]] = {}
    for frag in fragments:
        direction = str(frag.get("direction", "unknown"))
        by_direction.setdefault(direction, []).append(frag)

    all_records: List[Tuple[dict, dict, float, float]] = []
    for direction, frags in by_direction.items():
        if not frags:
            continue

        compute_nodes = sorted({str(frag.get("compute_node_id", "0")) for frag in frags})
        params = {
            "time_win": 15,
            "master_time_win": 20,
            "stitcher_mode": "local",
            "fragment_attr_name": "calib_id",
            "compute_node_list": compute_nodes if compute_nodes else [0],
            "stitcher_args": {
                "cx": 0.2,
                "mx": 0.1,
                "cy": 2,
                "my": 0.1,
                "stitch_thresh": 3,
                "master_stitch_thresh": 4,
            },
            "cost_function": {"type": "bhattacharyya"},
        }

        graph = MOTGraphSingle(direction=direction, attr=params["fragment_attr_name"], parameters=params)
        recorder = RecordingCostFunction(BhattacharyyaCostFunction())
        graph.cost_fn = recorder

        frags_sorted = sorted(frags, key=lambda x: x.get("first_timestamp", x["timestamp"][0]))
        for frag in frags_sorted:
            graph.add_node(frag)
            frag_id = frag[params["fragment_attr_name"]]
            graph.augment_path(frag_id)
            graph.pop_path(time_thresh=frag["first_timestamp"] - graph.TIME_WIN)

        all_records.extend(recorder.records)

    return all_records


def similarity_to_raw_base_cost(
    similarity: float,
    scale_factor: float,
    similarity_mapping: str,
    similarity_power: float,
    similarity_clip_eps: float,
) -> float:
    sim = float(np.clip(float(similarity), similarity_clip_eps, 1.0 - similarity_clip_eps))
    mapping = str(similarity_mapping).lower()
    if mapping == "odds":
        dissimilarity = (1.0 - sim) / sim
    elif mapping == "power":
        dissimilarity = (1.0 - sim) ** float(similarity_power)
    else:
        dissimilarity = (1.0 - sim)
    return float(float(scale_factor) * dissimilarity)


def _resolve_score_mapping(training_objective: str, score_mapping: str) -> str:
    objective = str(training_objective or "classification").lower()
    mapping = str(score_mapping or "auto").lower()
    if mapping == "auto":
        return "direct_cost" if objective == "ranking" else "legacy_similarity"
    return mapping


def _should_add_explicit_time_penalty(training_objective: str, score_mapping: str) -> bool:
    """
    Whether to add explicit `time_penalty * gap` outside the model output.

    Ranking + direct_cost models are trained against pipeline Bhattacharyya totals,
    so adding an extra explicit gap term at inference/eval double-counts time.
    """
    objective = str(training_objective or "classification").lower()
    mapping = _resolve_score_mapping(training_objective, score_mapping)
    return not (objective == "ranking" and mapping == "direct_cost")


def model_output_to_raw_base_cost(
    model_output: float,
    *,
    training_objective: str,
    score_mapping: str,
    scale_factor: float,
    similarity_mapping: str,
    similarity_power: float,
    similarity_clip_eps: float,
) -> float:
    mapping = _resolve_score_mapping(training_objective, score_mapping)
    if mapping == "direct_cost":
        x = float(model_output)
        return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))
    return similarity_to_raw_base_cost(
        float(model_output),
        scale_factor=scale_factor,
        similarity_mapping=similarity_mapping,
        similarity_power=similarity_power,
        similarity_clip_eps=similarity_clip_eps,
    )


@torch.no_grad()
def compute_transformer_raw_total(
    model: SiameseTransformerNetwork,
    seq_mean: torch.Tensor,
    seq_std: torch.Tensor,
    ep_mean: torch.Tensor,
    ep_std: torch.Tensor,
    device: torch.device,
    track1: dict,
    track2: dict,
    scale_factor: float,
    time_penalty: float,
    similarity_mapping: str,
    similarity_power: float,
    similarity_clip_eps: float,
    training_objective: str,
    score_mapping: str,
) -> float:
    seq_a = extract_rich_sequence(track1)
    seq_b = extract_rich_sequence(track2)
    endpoint = extract_endpoint_features(track1, track2)

    seq_a_t = torch.as_tensor(seq_a, dtype=torch.float32, device=device).unsqueeze(0)
    seq_b_t = torch.as_tensor(seq_b, dtype=torch.float32, device=device).unsqueeze(0)
    seq_a_t = (seq_a_t - seq_mean) / seq_std
    seq_b_t = (seq_b_t - seq_mean) / seq_std

    mask_a = torch.zeros(1, seq_a_t.size(1), dtype=torch.bool, device=device)
    mask_b = torch.zeros(1, seq_b_t.size(1), dtype=torch.bool, device=device)
    endpoint_t = torch.as_tensor(endpoint, dtype=torch.float32, device=device).unsqueeze(0)
    endpoint_t = (endpoint_t - ep_mean) / ep_std

    model_output = model(seq_a_t, mask_a, seq_b_t, mask_b, endpoint_t).item()
    gap = float(track2["timestamp"][0] - track1["timestamp"][-1])
    raw_base = model_output_to_raw_base_cost(
        model_output,
        training_objective=training_objective,
        score_mapping=score_mapping,
        scale_factor=scale_factor,
        similarity_mapping=similarity_mapping,
        similarity_power=similarity_power,
        similarity_clip_eps=similarity_clip_eps,
    )
    if _should_add_explicit_time_penalty(training_objective, score_mapping):
        return raw_base + time_penalty * gap
    return raw_base


def plot_calibration_distributions(raw_total: np.ndarray,
                                   calibrated_total: np.ndarray,
                                   bhat_total: np.ndarray,
                                   output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(bhat_total, bins=80, alpha=0.5, density=True, label="Bhattacharyya")
    axes[0].hist(raw_total, bins=80, alpha=0.5, density=True, label="Transformer raw")
    axes[0].hist(calibrated_total, bins=80, alpha=0.5, density=True, label="Transformer calibrated")
    axes[0].set_title("Cost Distribution Overlap")
    axes[0].set_xlabel("Cost")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for values, label in [
        (bhat_total, "Bhattacharyya"),
        (raw_total, "Transformer raw"),
        (calibrated_total, "Transformer calibrated"),
    ]:
        xs = np.sort(values)
        ys = np.linspace(0, 1, len(xs), endpoint=True)
        axes[1].plot(xs, ys, linewidth=2, label=label)
    axes[1].set_title("Cost CDF Comparison")
    axes[1].set_xlabel("Cost")
    axes[1].set_ylabel("CDF")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "transformer_calibration_distributions.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def run_pipeline_calibration(loaded: LoadedModel, args: argparse.Namespace, output_dir: Path):
    print(f"\nCollecting pipeline candidate pairs from scenario '{args.calibration_scenario}'...")
    records = collect_pipeline_pairs_for_calibration(args.calibration_scenario)
    print(f"Collected {len(records)} candidate pair evaluations from MOTGraphSingle.")

    if not records:
        raise RuntimeError("No calibration records collected from pipeline replay.")

    if args.max_calibration_pairs and len(records) > args.max_calibration_pairs:
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(records), size=args.max_calibration_pairs, replace=False)
        records = [records[int(i)] for i in chosen]
        print(f"Subsampled to {len(records)} calibration pairs (--max-calibration-pairs).")

    raw_total_vals = []
    bhat_total_vals = []
    model_device = next(loaded.model.parameters()).device

    for track1, track2, bhat_total, _gap in records:
        if not np.isfinite(bhat_total) or bhat_total >= 1e5:
            continue

        raw_total = compute_transformer_raw_total(
            loaded.model,
            loaded.seq_mean,
            loaded.seq_std,
            loaded.ep_mean,
            loaded.ep_std,
            model_device,
            track1,
            track2,
            args.scale_factor,
            args.time_penalty,
            args.similarity_mapping,
            args.similarity_power,
            args.similarity_clip_eps,
            args.training_objective,
            args.score_mapping,
        )

        if not np.isfinite(raw_total) or not np.isfinite(bhat_total):
            continue

        raw_total_vals.append(raw_total)
        bhat_total_vals.append(bhat_total)

    raw_total_np = np.asarray(raw_total_vals, dtype=np.float64)
    bhat_total_np = np.asarray(bhat_total_vals, dtype=np.float64)

    if len(raw_total_np) < 10:
        raise RuntimeError("Insufficient valid samples for calibration fitting.")

    fit_mask = np.isfinite(raw_total_np) & np.isfinite(bhat_total_np) & (bhat_total_np >= 0)
    if args.calibration_target_max > 0:
        fit_mask = fit_mask & (bhat_total_np <= args.calibration_target_max)

    if int(fit_mask.sum()) < 100:
        # Fallback: use robust clipping when explicit cap leaves too few points.
        cap = float(np.quantile(bhat_total_np[np.isfinite(bhat_total_np)], 0.95))
        fit_mask = np.isfinite(raw_total_np) & np.isfinite(bhat_total_np) & (bhat_total_np >= 0) & (bhat_total_np <= cap)

    if int(fit_mask.sum()) < 20:
        raise RuntimeError("Insufficient in-range samples after calibration target filtering.")

    raw_total_fit = raw_total_np[fit_mask]
    bhat_total_fit = bhat_total_np[fit_mask]

    artifact = fit_calibration(raw_total_fit, bhat_total_fit, args.calibration_mode)
    domain = [float(raw_total_np.min()), float(raw_total_np.max())]
    artifact["domain"] = domain
    artifact["fit_params"] = {
        "scale_factor": args.scale_factor,
        "time_penalty": args.time_penalty,
        "training_objective": args.training_objective,
        "score_mapping": args.score_mapping,
        "similarity_mapping": args.similarity_mapping,
        "similarity_power": args.similarity_power,
        "similarity_clip_eps": args.similarity_clip_eps,
        "scenario": args.calibration_scenario,
        "num_pairs": int(len(raw_total_np)),
        "num_fit_pairs": int(len(raw_total_fit)),
        "calibration_target_max": float(args.calibration_target_max),
        "target_mapping": "raw_total_to_bhat_total",
    }

    calibrated_total_fit = apply_calibration(raw_total_fit, artifact)

    artifact["distribution_overlap"] = {
        "raw_vs_bhat_mae_total": float(np.mean(np.abs(raw_total_fit - bhat_total_fit))),
        "calibrated_vs_bhat_mae_total": float(np.mean(np.abs(calibrated_total_fit - bhat_total_fit))),
    }

    calibration_path = Path(args.calibration_output)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibration_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved calibration artifact to: {calibration_path}")

    plot_calibration_distributions(raw_total_fit, calibrated_total_fit, bhat_total_fit, output_dir)
    print(f"Saved distribution overlap plot to: {output_dir / 'transformer_calibration_distributions.png'}")


def _safe_divide(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0 else 0.0


def _extract_fragment_id(fragment: dict, fallback: str) -> str:
    if "calib_id" in fragment and fragment["calib_id"] is not None:
        return str(fragment["calib_id"])

    if "local_fragment_id" in fragment and fragment["local_fragment_id"] is not None:
        lfid = fragment["local_fragment_id"]
        if isinstance(lfid, list) and lfid:
            return f"local_{lfid[0]}"
        return f"local_{lfid}"

    oid = fragment.get("_id")
    if isinstance(oid, dict) and "$oid" in oid:
        return str(oid["$oid"])
    if oid is not None:
        return str(oid)
    return str(fallback)


def _get_pair_gt_label(track1: dict, track2: dict) -> Optional[int]:
    gt1 = _get_gt_id(track1)
    gt2 = _get_gt_id(track2)
    if gt1 is None or gt2 is None:
        return None
    return int(gt1 == gt2)


def _build_anchor_outcomes(
    candidate_rows: List[Dict[str, Any]],
    min_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_anchor: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidate_rows:
        if row["gt_label"] is None:
            continue
        by_anchor.setdefault(row["anchor_id"], []).append(row)

    outcomes: List[Dict[str, Any]] = []
    for anchor_id, candidates in by_anchor.items():
        if len(candidates) < int(min_candidates):
            continue

        labels = [int(c["gt_label"]) for c in candidates]
        if 1 not in labels or 0 not in labels:
            continue

        bhat_idx = int(np.argmin([float(c["bhat_cost"]) for c in candidates]))
        tx_idx = int(np.argmin([float(c["tx_cost"]) for c in candidates]))

        pos_gaps = [float(c["gap"]) for c in candidates if int(c["gt_label"]) == 1]
        true_gap = float(min(pos_gaps)) if pos_gaps else float(min(float(c["gap"]) for c in candidates))

        outcomes.append({
            "anchor_id": str(anchor_id),
            "candidate_count": int(len(candidates)),
            "true_gap": true_gap,
            "bhat_correct": int(candidates[bhat_idx]["gt_label"]) == 1,
            "tx_correct": int(candidates[tx_idx]["gt_label"]) == 1,
            "bhat_pick_label": int(candidates[bhat_idx]["gt_label"]),
            "tx_pick_label": int(candidates[tx_idx]["gt_label"]),
            "bhat_pick_candidate_id": str(candidates[bhat_idx]["candidate_id"]),
            "tx_pick_candidate_id": str(candidates[tx_idx]["candidate_id"]),
        })

    return outcomes, by_anchor


def _compute_anchor_metrics(outcomes: List[Dict[str, Any]], include_anchor_ids: bool = False) -> Dict[str, Any]:
    total = int(len(outcomes))
    bhat_wrong = [o for o in outcomes if not bool(o["bhat_correct"])]
    tx_wrong = [o for o in outcomes if not bool(o["tx_correct"])]

    bhat_wrong_count = int(len(bhat_wrong))
    tx_wrong_count = int(len(tx_wrong))
    bhat_correct_count = int(total - bhat_wrong_count)

    recovered = [o for o in outcomes if (not bool(o["bhat_correct"])) and bool(o["tx_correct"])]
    regressions = [o for o in outcomes if bool(o["bhat_correct"]) and (not bool(o["tx_correct"]))]

    bhat_top1_error_rate = _safe_divide(bhat_wrong_count, total)
    tx_top1_error_rate = _safe_divide(tx_wrong_count, total)

    if bhat_wrong_count == 0:
        edge_error_lift = 0.0
    else:
        edge_error_lift = _safe_divide((bhat_top1_error_rate - tx_top1_error_rate), bhat_top1_error_rate)

    metrics: Dict[str, Any] = {
        "num_anchors": total,
        "bhat_top1_error_count": bhat_wrong_count,
        "tx_top1_error_count": tx_wrong_count,
        "bhat_top1_error_rate": float(bhat_top1_error_rate),
        "tx_top1_error_rate": float(tx_top1_error_rate),
        "edge_error_lift": float(edge_error_lift),
        "bhat_failure_recovery_count": int(len(recovered)),
        "bhat_failure_recovery_rate": float(_safe_divide(len(recovered), bhat_wrong_count)),
        "tx_regression_on_bhat_correct_count": int(len(regressions)),
        "tx_regression_rate_on_bhat_correct": float(_safe_divide(len(regressions), bhat_correct_count)),
    }
    if include_anchor_ids:
        metrics["bhat_wrong_anchor_ids"] = [str(o["anchor_id"]) for o in bhat_wrong]
        metrics["tx_wrong_anchor_ids"] = [str(o["anchor_id"]) for o in tx_wrong]
        metrics["bhat_wrong_tx_right_anchor_ids"] = [str(o["anchor_id"]) for o in recovered]
        metrics["bhat_right_tx_wrong_anchor_ids"] = [str(o["anchor_id"]) for o in regressions]
    return metrics


def _compute_pair_binary_metrics(candidate_rows: List[Dict[str, Any]], cost_key: str) -> Dict[str, Any]:
    labeled_rows = [row for row in candidate_rows if row["gt_label"] is not None]
    if not labeled_rows:
        return {"num_pairs": 0, "roc_auc": None, "avg_precision": None}

    labels = np.asarray([int(row["gt_label"]) for row in labeled_rows], dtype=np.int64)
    costs = np.asarray([float(row[cost_key]) for row in labeled_rows], dtype=np.float64)

    if len(np.unique(labels)) < 2:
        return {"num_pairs": int(len(labels)), "roc_auc": None, "avg_precision": None}

    scores = -costs  # lower cost => better match
    try:
        roc_auc = float(roc_auc_score(labels, scores))
    except ValueError:
        roc_auc = None
    try:
        avg_precision = float(average_precision_score(labels, scores))
    except ValueError:
        avg_precision = None
    return {
        "num_pairs": int(len(labels)),
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
    }


def _bucket_gap(true_gap: float) -> str:
    gap = float(true_gap)
    if gap <= 2.0:
        return "0-2"
    if gap <= 5.0:
        return "2-5"
    if gap <= 15.0:
        return "5-15"
    return "other"


def _bucket_candidate_count(candidate_count: int) -> str:
    n = int(candidate_count)
    if n <= 2:
        return "2"
    if n <= 4:
        return "3-4"
    return "5+"


def _slice_anchor_metrics(
    outcomes: List[Dict[str, Any]],
    bucket_fn,
    bucket_order: List[str],
) -> Dict[str, Dict[str, Any]]:
    sliced: Dict[str, Dict[str, Any]] = {}
    for bucket in bucket_order:
        subset = [o for o in outcomes if bucket_fn(o) == bucket]
        sliced[bucket] = _compute_anchor_metrics(subset, include_anchor_ids=False)
    return sliced


def _collect_error_examples(
    by_anchor: Dict[str, List[Dict[str, Any]]],
    anchor_ids: List[str],
    limit: int = 100,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for anchor_id in anchor_ids:
        for row in by_anchor.get(anchor_id, []):
            examples.append({
                "anchor_id": str(anchor_id),
                "candidate_id": str(row["candidate_id"]),
                "bhat_cost": float(row["bhat_cost"]),
                "transformer_cost": float(row["tx_cost"]),
                "gt_label": int(row["gt_label"]),
                "gap": float(row["gap"]),
            })
            if len(examples) >= int(limit):
                return examples
    return examples


def _make_complementarity_decision(overall: Dict[str, Any], go_lift_threshold: float) -> Dict[str, Any]:
    num_anchors = int(overall.get("num_anchors", 0))
    if num_anchors <= 0:
        return {
            "recommendation": "pivot",
            "checks": [],
            "reasons": [
                "insufficient_data: no eligible anchors after GT/min-candidate filtering",
            ],
        }

    recovery = float(overall["bhat_failure_recovery_rate"])
    regression = float(overall["tx_regression_rate_on_bhat_correct"])
    tx_error = float(overall["tx_top1_error_rate"])
    bhat_error = float(overall["bhat_top1_error_rate"])

    checks = [
        {
            "name": "recovery_rate",
            "pass": bool(recovery >= float(go_lift_threshold)),
            "actual": recovery,
            "threshold": float(go_lift_threshold),
            "operator": ">=",
        },
        {
            "name": "regression_rate",
            "pass": bool(regression <= 0.05),
            "actual": regression,
            "threshold": 0.05,
            "operator": "<=",
        },
        {
            "name": "transformer_vs_bhat_error",
            "pass": bool(tx_error <= bhat_error),
            "actual": tx_error,
            "threshold": bhat_error,
            "operator": "<=",
        },
    ]
    all_pass = all(check["pass"] for check in checks)
    reasons = [
        f"{check['name']}: {'PASS' if check['pass'] else 'FAIL'} "
        f"({check['actual']:.4f} {check['operator']} {check['threshold']:.4f})"
        for check in checks
    ]
    return {
        "recommendation": "continue" if all_pass else "pivot",
        "checks": checks,
        "reasons": reasons,
    }


def plot_complementarity_anchor_errors(overall: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Bhattacharyya", "Transformer"]
    values = [
        float(overall["bhat_top1_error_rate"]),
        float(overall["tx_top1_error_rate"]),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top-1 Anchor Error Rate")
    ax.set_title("Anchor-Level Error Comparison")
    ax.grid(axis="y", alpha=0.3)
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    out_path = output_dir / "transformer_complementarity_anchor_error_bar.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_complementarity_gap_slices(slice_metrics: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    gap_order = ["0-2", "2-5", "5-15"]
    bhat_vals = [float(slice_metrics["gap_bins"][bucket]["bhat_top1_error_rate"]) for bucket in gap_order]
    tx_vals = [float(slice_metrics["gap_bins"][bucket]["tx_top1_error_rate"]) for bucket in gap_order]

    x = np.arange(len(gap_order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, bhat_vals, width, label="Bhattacharyya", color="#1f77b4")
    ax.bar(x + width / 2, tx_vals, width, label="Transformer", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(gap_order)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("True-Match Gap Bin (s)")
    ax.set_ylabel("Top-1 Anchor Error Rate")
    ax.set_title("Error Rate by Gap Slice")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "transformer_complementarity_gap_slices.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_complementarity_audit(loaded: LoadedModel, args: argparse.Namespace, output_dir: Path):
    print(f"\nCollecting pipeline candidate pairs from scenario '{args.audit_scenario}' for complementarity audit...")
    records = collect_pipeline_pairs_for_calibration(args.audit_scenario)
    print(f"Collected {len(records)} candidate pair evaluations from MOTGraphSingle.")
    if not records:
        raise RuntimeError("No records collected from pipeline replay for complementarity audit.")

    if args.audit_max_pairs and len(records) > args.audit_max_pairs:
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(records), size=args.audit_max_pairs, replace=False)
        records = [records[int(i)] for i in chosen]
        print(f"Subsampled to {len(records)} audit pairs (--audit-max-pairs).")

    model_device = next(loaded.model.parameters()).device
    candidate_rows: List[Dict[str, Any]] = []

    for idx, (track1, track2, bhat_total, gap) in enumerate(records):
        if not np.isfinite(bhat_total) or float(bhat_total) >= 1e5:
            continue

        tx_total = compute_transformer_raw_total(
            loaded.model,
            loaded.seq_mean,
            loaded.seq_std,
            loaded.ep_mean,
            loaded.ep_std,
            model_device,
            track1,
            track2,
            args.scale_factor,
            args.time_penalty,
            args.similarity_mapping,
            args.similarity_power,
            args.similarity_clip_eps,
            args.training_objective,
            args.score_mapping,
        )
        if not np.isfinite(tx_total):
            continue

        anchor_id = _extract_fragment_id(track2, f"anchor_{idx}")
        candidate_id = _extract_fragment_id(track1, f"candidate_{idx}")
        candidate_rows.append({
            "anchor_id": str(anchor_id),
            "candidate_id": str(candidate_id),
            "bhat_cost": float(bhat_total),
            "tx_cost": float(tx_total),
            "gap": float(gap),
            "gt_label": _get_pair_gt_label(track1, track2),
        })

    if not candidate_rows:
        raise RuntimeError("No valid candidate rows available after cost filtering for complementarity audit.")

    outcomes, by_anchor = _build_anchor_outcomes(candidate_rows, args.audit_min_candidates)
    overall_metrics = _compute_anchor_metrics(outcomes, include_anchor_ids=True)
    if overall_metrics["num_anchors"] <= 0:
        print(
            "Warning: no eligible anchors after GT/min-candidate filtering. "
            "Emitting audit artifact with insufficient-data decision."
        )

    pair_metrics = {
        "bhat": _compute_pair_binary_metrics(candidate_rows, cost_key="bhat_cost"),
        "transformer": _compute_pair_binary_metrics(candidate_rows, cost_key="tx_cost"),
    }

    slice_metrics = {
        "gap_bins": _slice_anchor_metrics(
            outcomes,
            bucket_fn=lambda o: _bucket_gap(float(o["true_gap"])),
            bucket_order=["0-2", "2-5", "5-15", "other"],
        ),
        "candidate_count_bins": _slice_anchor_metrics(
            outcomes,
            bucket_fn=lambda o: _bucket_candidate_count(int(o["candidate_count"])),
            bucket_order=["2", "3-4", "5+"],
        ),
    }

    recovered_anchor_ids = sorted(set(overall_metrics["bhat_wrong_tx_right_anchor_ids"]))
    regression_anchor_ids = sorted(set(overall_metrics["bhat_right_tx_wrong_anchor_ids"]))
    decision = _make_complementarity_decision(overall_metrics, go_lift_threshold=args.audit_go_lift_threshold)

    audit_plots_dir = Path(args.audit_plots_dir)
    anchor_plot_path = plot_complementarity_anchor_errors(overall_metrics, audit_plots_dir)
    gap_plot_path = plot_complementarity_gap_slices(slice_metrics, audit_plots_dir)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(loaded.checkpoint_path),
        "audit_scenario": str(args.audit_scenario),
        "num_records_collected": int(len(records)),
        "num_candidate_rows_valid_costs": int(len(candidate_rows)),
        "num_candidate_rows_with_gt": int(sum(row["gt_label"] is not None for row in candidate_rows)),
        "num_anchors_with_known_gt": int(len(by_anchor)),
        "eligible_anchor_count": int(overall_metrics["num_anchors"]),
        "params": {
            "audit_min_candidates": int(args.audit_min_candidates),
            "audit_go_lift_threshold": float(args.audit_go_lift_threshold),
            "scale_factor": float(args.scale_factor),
            "time_penalty": float(args.time_penalty),
            "training_objective": str(args.training_objective),
            "score_mapping": str(args.score_mapping),
            "similarity_mapping": str(args.similarity_mapping),
            "similarity_power": float(args.similarity_power),
            "similarity_clip_eps": float(args.similarity_clip_eps),
            "audit_max_pairs": int(args.audit_max_pairs),
        },
        "plots": {
            "anchor_error_bar": str(anchor_plot_path),
            "gap_slices": str(gap_plot_path),
        },
    }

    artifact = {
        "metadata": metadata,
        "overall_metrics": overall_metrics,
        "pair_level_metrics": pair_metrics,
        "slice_metrics": slice_metrics,
        "error_sets": {
            "bhat_wrong_tx_right_anchor_ids": recovered_anchor_ids,
            "bhat_right_tx_wrong_anchor_ids": regression_anchor_ids,
            "bhat_wrong_tx_right_examples": _collect_error_examples(by_anchor, recovered_anchor_ids, limit=100),
            "bhat_right_tx_wrong_examples": _collect_error_examples(by_anchor, regression_anchor_ids, limit=100),
        },
        "decision": decision,
    }

    audit_output_path = Path(args.audit_output)
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved complementarity audit to: {audit_output_path}")
    print(f"Saved complementarity plots to: {audit_plots_dir}")
    print(
        "Audit decision: "
        f"{decision['recommendation']} | "
        f"recovery={overall_metrics['bhat_failure_recovery_rate']:.4f}, "
        f"regression={overall_metrics['tx_regression_rate_on_bhat_correct']:.4f}, "
        f"bhat_err={overall_metrics['bhat_top1_error_rate']:.4f}, "
        f"tx_err={overall_metrics['tx_top1_error_rate']:.4f}"
    )


def _compute_ranking_gate_metrics(candidate_rows: List[Dict[str, Any]], min_candidates: int) -> Dict[str, Any]:
    by_anchor: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidate_rows:
        if row["gt_label"] is None:
            continue
        by_anchor.setdefault(str(row["anchor_id"]), []).append(row)

    eligible = 0
    top1_correct = 0
    pairwise_correct = 0
    pairwise_total = 0
    spearman_vals: List[float] = []

    for _anchor_id, candidates in by_anchor.items():
        if len(candidates) < int(min_candidates):
            continue
        labels = np.asarray([int(c["gt_label"]) for c in candidates], dtype=np.int64)
        if 1 not in labels or 0 not in labels:
            continue

        tx_costs = np.asarray([float(c["tx_cost"]) for c in candidates], dtype=np.float64)
        bhat_costs = np.asarray([float(c["bhat_cost"]) for c in candidates], dtype=np.float64)
        if not np.all(np.isfinite(tx_costs)) or not np.all(np.isfinite(bhat_costs)):
            continue

        eligible += 1

        top1_idx = int(np.argmin(tx_costs))
        top1_correct += int(labels[top1_idx] == 1)

        pos_costs = tx_costs[labels == 1]
        neg_costs = tx_costs[labels == 0]
        if len(pos_costs) > 0 and len(neg_costs) > 0:
            pairwise_total += int(len(pos_costs) * len(neg_costs))
            pairwise_correct += int((pos_costs[:, None] < neg_costs[None, :]).sum())

        spearman_vals.append(_spearman_corr(tx_costs, bhat_costs))

    return {
        "eligible_anchor_count": int(eligible),
        "pairwise_acc_gt_crossclass": float(_safe_divide(pairwise_correct, pairwise_total)),
        "anchor_top1_acc": float(_safe_divide(top1_correct, eligible)),
        "mean_anchor_spearman": float(np.mean(spearman_vals)) if spearman_vals else 0.0,
        "pairwise_correct": int(pairwise_correct),
        "pairwise_total": int(pairwise_total),
        "top1_correct": int(top1_correct),
        "top1_total": int(eligible),
        "spearman_anchor_count": int(len(spearman_vals)),
    }


def _make_ranking_gate_decision(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    checks = [
        {
            "name": "pairwise_acc_gt_crossclass",
            "actual": float(metrics["pairwise_acc_gt_crossclass"]),
            "threshold": float(args.gate_pairwise_threshold),
            "operator": ">=",
        },
        {
            "name": "anchor_top1_acc",
            "actual": float(metrics["anchor_top1_acc"]),
            "threshold": float(args.gate_top1_threshold),
            "operator": ">=",
        },
        {
            "name": "mean_anchor_spearman",
            "actual": float(metrics["mean_anchor_spearman"]),
            "threshold": float(args.gate_spearman_threshold),
            "operator": ">=",
        },
    ]
    for check in checks:
        check["pass"] = bool(check["actual"] >= check["threshold"])
    decision = "promote_to_pipeline_sanity_check" if all(c["pass"] for c in checks) else "hold"
    reasons = [
        f"{c['name']}: {'PASS' if c['pass'] else 'FAIL'} "
        f"({c['actual']:.4f} {c['operator']} {c['threshold']:.4f})"
        for c in checks
    ]
    return {"recommendation": decision, "checks": checks, "reasons": reasons}


def _resolve_ranking_dataset_path(loaded: LoadedModel, args: argparse.Namespace) -> Path:
    raw_path = str(args.dataset_path or "").strip()
    if not raw_path:
        raw_path = str(loaded.train_config.get("dataset_path", "")).strip()
    if not raw_path:
        return PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_dataset.jsonl"

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return PROJECT_ROOT / path


def _resolve_ranking_output_path(args: argparse.Namespace) -> Path:
    if str(args.ranking_output or "").strip():
        return Path(args.ranking_output)

    output_dir = PROJECT_ROOT / "models" / "outputs"
    if bool(args.replay):
        scenario = str(args.ranking_scenario).strip() or "i"
        return output_dir / f"transformer_ranking_gate_{scenario}.json"
    split = str(args.ranking_split).strip() or "val"
    return output_dir / f"transformer_ranking_gate_{split}.json"


def _load_fragment_store(fragments_path: Path) -> Dict[str, Dict[str, Any]]:
    if not fragments_path.exists():
        raise FileNotFoundError(
            f"Fragment store not found for ranking dataset refs: {fragments_path}"
        )

    store: Dict[str, Dict[str, Any]] = {}
    with open(fragments_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ref = row.get("fragment_ref")
            frag = row.get("fragment")
            if ref is None or not isinstance(frag, dict):
                continue
            store[str(ref)] = frag
    return store


def _resolve_dataset_row_tracks(
    row: Dict[str, Any],
    fragment_store: Dict[str, Dict[str, Any]],
    fragments_path: Path,
) -> Tuple[dict, dict]:
    track_candidate = row.get("track_candidate")
    track_anchor = row.get("track_anchor")
    if isinstance(track_candidate, dict) and isinstance(track_anchor, dict):
        return track_candidate, track_anchor

    candidate_ref = row.get("candidate_ref")
    anchor_ref = row.get("anchor_ref")
    if candidate_ref is None or anchor_ref is None:
        raise KeyError(
            "Row is missing embedded tracks and fragment refs (candidate_ref/anchor_ref)."
        )

    try:
        return (
            fragment_store[str(candidate_ref)],
            fragment_store[str(anchor_ref)],
        )
    except KeyError as exc:
        raise KeyError(
            f"Missing fragment_ref '{exc.args[0]}' in fragment store {fragments_path}"
        ) from exc


def _load_candidate_rows_from_dataset(
    loaded: LoadedModel,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset_path = _resolve_ranking_dataset_path(loaded, args)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Ranking dataset not found: {dataset_path}")

    print(
        "\nLoading ranking-gate rows from dataset "
        f"'{dataset_path}' (split={args.ranking_split})..."
    )

    selected_rows: List[Dict[str, Any]] = []
    dataset_total_rows = 0
    dataset_split_rows = 0
    split_filter = str(args.ranking_split)
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dataset_total_rows += 1
            row = json.loads(line)
            row_split = str(row.get("split", "train"))
            if split_filter != "all" and row_split != split_filter:
                continue
            dataset_split_rows += 1
            selected_rows.append(row)

    if not selected_rows:
        raise RuntimeError(
            f"No rows found in dataset split '{split_filter}' from {dataset_path}."
        )

    if args.ranking_max_pairs and len(selected_rows) > args.ranking_max_pairs:
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(selected_rows), size=args.ranking_max_pairs, replace=False)
        selected_rows = [selected_rows[int(i)] for i in chosen]
        print(f"Subsampled to {len(selected_rows)} ranking rows (--ranking-max-pairs).")

    fragments_path = dataset_path.parent / f"{dataset_path.stem}.fragments.jsonl"
    needs_fragment_store = any(
        not (isinstance(row.get("track_candidate"), dict) and isinstance(row.get("track_anchor"), dict))
        for row in selected_rows
    )
    fragment_store: Dict[str, Dict[str, Any]] = {}
    if needs_fragment_store:
        fragment_store = _load_fragment_store(fragments_path)

    model_device = next(loaded.model.parameters()).device
    candidate_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(selected_rows):
        bhat_total = row.get("bhat_cost")
        if bhat_total is None or not np.isfinite(float(bhat_total)) or float(bhat_total) >= 1e5:
            continue

        track1, track2 = _resolve_dataset_row_tracks(row, fragment_store, fragments_path)
        tx_total = compute_transformer_raw_total(
            loaded.model,
            loaded.seq_mean,
            loaded.seq_std,
            loaded.ep_mean,
            loaded.ep_std,
            model_device,
            track1,
            track2,
            args.scale_factor,
            args.time_penalty,
            args.similarity_mapping,
            args.similarity_power,
            args.similarity_clip_eps,
            args.training_objective,
            args.score_mapping,
        )
        if not np.isfinite(tx_total):
            continue

        gt_raw = row.get("gt_label")
        gt_label = int(gt_raw) if gt_raw in (0, 1) else None
        gap = row.get("gap")
        if gap is None or not np.isfinite(float(gap)):
            gap = float(track2["timestamp"][0] - track1["timestamp"][-1])

        anchor_group_id = row.get("anchor_key")
        if anchor_group_id is None:
            anchor_group_id = row.get("anchor_id")
        if anchor_group_id is None:
            anchor_group_id = _extract_fragment_id(track2, f"anchor_{idx}")

        candidate_id = row.get("candidate_id")
        if candidate_id is None:
            candidate_id = _extract_fragment_id(track1, f"candidate_{idx}")

        candidate_rows.append(
            {
                "anchor_id": str(anchor_group_id),
                "candidate_id": str(candidate_id),
                "bhat_cost": float(bhat_total),
                "tx_cost": float(tx_total),
                "gap": float(gap),
                "gt_label": gt_label,
            }
        )

    meta: Dict[str, Any] = {
        "data_source": "dataset",
        "dataset_path": str(dataset_path),
        "ranking_split": str(split_filter),
        "num_dataset_rows_total": int(dataset_total_rows),
        "num_dataset_rows_split_selected": int(dataset_split_rows),
        "num_dataset_rows_after_cap": int(len(selected_rows)),
    }
    if needs_fragment_store:
        meta["fragments_path"] = str(fragments_path)
    return candidate_rows, meta


def _load_candidate_rows_from_replay(
    loaded: LoadedModel,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"\nCollecting pipeline candidate pairs from scenario '{args.ranking_scenario}' for ranking-gate eval...")
    records = collect_pipeline_pairs_for_calibration(args.ranking_scenario)
    print(f"Collected {len(records)} candidate pair evaluations from MOTGraphSingle.")
    if not records:
        raise RuntimeError("No records collected from pipeline replay for ranking gate evaluation.")

    if args.ranking_max_pairs and len(records) > args.ranking_max_pairs:
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(records), size=args.ranking_max_pairs, replace=False)
        records = [records[int(i)] for i in chosen]
        print(f"Subsampled to {len(records)} ranking pairs (--ranking-max-pairs).")

    model_device = next(loaded.model.parameters()).device
    candidate_rows: List[Dict[str, Any]] = []

    for idx, (track1, track2, bhat_total, gap) in enumerate(records):
        if not np.isfinite(bhat_total) or float(bhat_total) >= 1e5:
            continue

        tx_total = compute_transformer_raw_total(
            loaded.model,
            loaded.seq_mean,
            loaded.seq_std,
            loaded.ep_mean,
            loaded.ep_std,
            model_device,
            track1,
            track2,
            args.scale_factor,
            args.time_penalty,
            args.similarity_mapping,
            args.similarity_power,
            args.similarity_clip_eps,
            args.training_objective,
            args.score_mapping,
        )
        if not np.isfinite(tx_total):
            continue

        candidate_rows.append(
            {
                "anchor_id": _extract_fragment_id(track2, f"anchor_{idx}"),
                "candidate_id": _extract_fragment_id(track1, f"candidate_{idx}"),
                "bhat_cost": float(bhat_total),
                "tx_cost": float(tx_total),
                "gap": float(gap),
                "gt_label": _get_pair_gt_label(track1, track2),
            }
        )

    meta: Dict[str, Any] = {
        "data_source": "replay",
        "ranking_scenario": str(args.ranking_scenario),
        "num_records_collected": int(len(records)),
    }
    return candidate_rows, meta


def run_ranking_gate_eval(loaded: LoadedModel, args: argparse.Namespace):
    if bool(args.replay):
        candidate_rows, source_meta = _load_candidate_rows_from_replay(loaded, args)
    else:
        candidate_rows, source_meta = _load_candidate_rows_from_dataset(loaded, args)

    if not candidate_rows:
        source_label = "replay" if bool(args.replay) else "dataset"
        raise RuntimeError(
            f"No valid candidate rows available after cost filtering for ranking gate evaluation ({source_label})."
        )

    metrics = _compute_ranking_gate_metrics(candidate_rows, min_candidates=args.ranking_min_candidates)
    decision = _make_ranking_gate_decision(metrics, args)

    artifact = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(loaded.checkpoint_path),
            **source_meta,
            "num_candidate_rows_valid_costs": int(len(candidate_rows)),
            "num_candidate_rows_with_gt": int(sum(row["gt_label"] is not None for row in candidate_rows)),
            "params": {
                "ranking_min_candidates": int(args.ranking_min_candidates),
                "scale_factor": float(args.scale_factor),
                "time_penalty": float(args.time_penalty),
                "training_objective": str(args.training_objective),
                "score_mapping": str(args.score_mapping),
                "similarity_mapping": str(args.similarity_mapping),
                "similarity_power": float(args.similarity_power),
                "similarity_clip_eps": float(args.similarity_clip_eps),
                "ranking_max_pairs": int(args.ranking_max_pairs),
                "replay": bool(args.replay),
                "ranking_split": str(args.ranking_split),
            },
            "gate_thresholds": {
                "pairwise_acc_gt_crossclass": float(args.gate_pairwise_threshold),
                "anchor_top1_acc": float(args.gate_top1_threshold),
                "mean_anchor_spearman": float(args.gate_spearman_threshold),
            },
        },
        "metrics": metrics,
        "decision": decision,
    }

    out_path = _resolve_ranking_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved ranking gate evaluation to: {out_path}")
    print(
        "Ranking gate decision: "
        f"{decision['recommendation']} | "
        f"pairwise={metrics['pairwise_acc_gt_crossclass']:.4f}, "
        f"top1={metrics['anchor_top1_acc']:.4f}, "
        f"spearman={metrics['mean_anchor_spearman']:.4f}"
    )


def main():
    args = parse_args()
    args.training_objective = str(args.training_objective).lower()
    args.score_mapping = str(args.score_mapping).lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded = load_model(checkpoint_path, device)
    checkpoint_objective = str(loaded.model_config.get("training_objective", args.training_objective)).lower()
    if checkpoint_objective in {"classification", "ranking"} and checkpoint_objective != args.training_objective:
        print(
            "Info: overriding --training-objective from checkpoint model_config "
            f"({args.training_objective} -> {checkpoint_objective})."
        )
        args.training_objective = checkpoint_objective
    args.score_mapping = _resolve_score_mapping(args.training_objective, args.score_mapping)
    print(f"Loaded transformer checkpoint: {loaded.checkpoint_path}")
    print(f"Model config: {loaded.model_config}")
    print(f"Using device: {device}")
    print(f"Objective mapping: training_objective={args.training_objective}, score_mapping={args.score_mapping}")

    if args.audit_complementarity:
        run_complementarity_audit(loaded, args, output_dir)
        return

    if args.evaluate_ranking_gates:
        run_ranking_gate_eval(loaded, args)
        return

    dataset = RichSequenceDataset(dataset_names=args.dataset_names, normalize=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=rich_collate_fn,
    )

    metrics = evaluate_classification(
        loaded.model,
        loader,
        device,
        loaded.seq_mean,
        loaded.seq_std,
        loaded.ep_mean,
        loaded.ep_std,
    )
    print("\nEvaluation Metrics:")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  AP:      {metrics['avg_precision']:.4f}")
    print(f"  Acc:     {metrics['accuracy']:.4f}")

    plot_standard_metrics(metrics, output_dir)
    print(f"Saved evaluation plots to: {output_dir / 'transformer_evaluation.png'}")

    serializable_metrics = {
        "roc_auc": metrics["roc_auc"],
        "avg_precision": metrics["avg_precision"],
        "accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }
    with open(output_dir / "transformer_eval_metrics.json", "w") as f:
        json.dump(serializable_metrics, f, indent=2, default=float)

    if args.export_attention:
        export_attention_maps(
            loaded.model,
            loader,
            device,
            loaded.seq_mean,
            loaded.seq_std,
            loaded.ep_mean,
            loaded.ep_std,
            output_dir=output_dir,
            max_pairs=args.max_attention_pairs,
        )

    if args.fit_calibration:
        run_pipeline_calibration(loaded, args, output_dir)


if __name__ == "__main__":
    main()
