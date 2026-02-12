"""
Evaluation and Calibration Script for Siamese Transformer Model

This script provides:
1. Standard classification evaluation (ROC-AUC, AP, accuracy, confusion matrix)
2. Attention map export for explainability
3. Pipeline-based cost calibration against Bhattacharyya cost distribution
"""

import argparse
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    model_config: Dict
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
        help="Scale factor used in raw transformer base cost: (1-similarity)*scale_factor.",
    )
    parser.add_argument(
        "--time-penalty",
        type=float,
        default=0.1,
        help="Time penalty used in runtime total cost calculation.",
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
    )

    return LoadedModel(
        model=model,
        seq_mean=seq_mean,
        seq_std=seq_std,
        model_config=model_config,
        checkpoint_path=checkpoint_path,
    )


@torch.no_grad()
def evaluate_classification(model: SiameseTransformerNetwork,
                            loader: DataLoader,
                            device: torch.device,
                            seq_mean: torch.Tensor,
                            seq_std: torch.Tensor) -> Dict:
    preds: List[float] = []
    labels: List[float] = []

    for pa, _la, ma, pb, _lb, mb, eps, y in loader:
        pa = pa.to(device)
        ma = ma.to(device)
        pb = pb.to(device)
        mb = mb.to(device)
        eps = eps.to(device)
        pa = (pa - seq_mean) / (seq_std + 1e-8)
        pb = (pb - seq_mean) / (seq_std + 1e-8)

        prob = model(pa, ma, pb, mb, eps).squeeze(1)
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
        pa = (pa - seq_mean) / (seq_std + 1e-8)
        pb = (pb - seq_mean) / (seq_std + 1e-8)
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


@torch.no_grad()
def compute_transformer_raw_base(
    model: SiameseTransformerNetwork,
    seq_mean: torch.Tensor,
    seq_std: torch.Tensor,
    device: torch.device,
    track1: dict,
    track2: dict,
    scale_factor: float,
) -> float:
    seq_a = extract_rich_sequence(track1)
    seq_b = extract_rich_sequence(track2)
    endpoint = extract_endpoint_features(track1, track2)

    seq_a_t = torch.as_tensor(seq_a, dtype=torch.float32, device=device).unsqueeze(0)
    seq_b_t = torch.as_tensor(seq_b, dtype=torch.float32, device=device).unsqueeze(0)
    seq_a_t = (seq_a_t - seq_mean) / (seq_std + 1e-8)
    seq_b_t = (seq_b_t - seq_mean) / (seq_std + 1e-8)

    mask_a = torch.zeros(1, seq_a_t.size(1), dtype=torch.bool, device=device)
    mask_b = torch.zeros(1, seq_b_t.size(1), dtype=torch.bool, device=device)
    endpoint_t = torch.as_tensor(endpoint, dtype=torch.float32, device=device).unsqueeze(0)

    similarity = float(model(seq_a_t, mask_a, seq_b_t, mask_b, endpoint_t).item())
    return (1.0 - similarity) * scale_factor


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

    raw_base_vals = []
    bhat_base_vals = []
    gaps = []
    bhat_total_vals = []
    model_device = next(loaded.model.parameters()).device

    for track1, track2, bhat_total, gap in records:
        if not np.isfinite(bhat_total) or bhat_total >= 1e5:
            continue

        raw_base = compute_transformer_raw_base(
            loaded.model,
            loaded.seq_mean,
            loaded.seq_std,
            model_device,
            track1,
            track2,
            args.scale_factor,
        )
        bhat_base = bhat_total - args.time_penalty * gap

        if not np.isfinite(raw_base) or not np.isfinite(bhat_base):
            continue

        raw_base_vals.append(raw_base)
        bhat_base_vals.append(bhat_base)
        gaps.append(gap)
        bhat_total_vals.append(bhat_total)

    raw_base_np = np.asarray(raw_base_vals, dtype=np.float64)
    bhat_base_np = np.asarray(bhat_base_vals, dtype=np.float64)
    gaps_np = np.asarray(gaps, dtype=np.float64)
    bhat_total_np = np.asarray(bhat_total_vals, dtype=np.float64)

    if len(raw_base_np) < 10:
        raise RuntimeError("Insufficient valid samples for calibration fitting.")

    fit_mask = np.isfinite(raw_base_np) & np.isfinite(bhat_base_np) & (bhat_base_np >= 0)
    if args.calibration_target_max > 0:
        fit_mask = fit_mask & (bhat_base_np <= args.calibration_target_max)

    if int(fit_mask.sum()) < 100:
        # Fallback: use robust clipping when explicit cap leaves too few points.
        cap = float(np.quantile(bhat_base_np[np.isfinite(bhat_base_np)], 0.95))
        fit_mask = np.isfinite(raw_base_np) & np.isfinite(bhat_base_np) & (bhat_base_np >= 0) & (bhat_base_np <= cap)

    if int(fit_mask.sum()) < 20:
        raise RuntimeError("Insufficient in-range samples after calibration target filtering.")

    raw_base_fit = raw_base_np[fit_mask]
    bhat_base_fit = bhat_base_np[fit_mask]
    gaps_fit = gaps_np[fit_mask]
    bhat_total_fit = bhat_total_np[fit_mask]

    artifact = fit_calibration(raw_base_fit, bhat_base_fit, args.calibration_mode)
    domain = [float(raw_base_np.min()), float(raw_base_np.max())]
    artifact["domain"] = domain
    artifact["fit_params"] = {
        "scale_factor": args.scale_factor,
        "time_penalty": args.time_penalty,
        "scenario": args.calibration_scenario,
        "num_pairs": int(len(raw_base_np)),
        "num_fit_pairs": int(len(raw_base_fit)),
        "calibration_target_max": float(args.calibration_target_max),
    }

    calibrated_base_fit = apply_calibration(raw_base_fit, artifact)
    raw_total_fit = raw_base_fit + args.time_penalty * gaps_fit
    calibrated_total_fit = calibrated_base_fit + args.time_penalty * gaps_fit

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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded = load_model(checkpoint_path, device)
    print(f"Loaded transformer checkpoint: {loaded.checkpoint_path}")
    print(f"Model config: {loaded.model_config}")
    print(f"Using device: {device}")

    dataset = RichSequenceDataset(dataset_names=args.dataset_names, normalize=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=rich_collate_fn,
    )

    metrics = evaluate_classification(loaded.model, loader, device, loaded.seq_mean, loaded.seq_std)
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
            output_dir=output_dir,
            max_pairs=args.max_attention_pairs,
        )

    if args.fit_calibration:
        run_pipeline_calibration(loaded, args, output_dir)


if __name__ == "__main__":
    main()
