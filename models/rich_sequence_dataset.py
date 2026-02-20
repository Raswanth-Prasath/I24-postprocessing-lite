"""
Rich 8-Feature Sequence Dataset for TCN and Transformer Models

Extracts per-timestep features: [x_position, y_position, velocity, length, width, height,
detection_confidence, time_normalized] from raw trajectory fragments.

Shared by TCN and Transformer training pipelines.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
from bisect import bisect_left, bisect_right


NUM_FEATURES = 8  # x, y, velocity, length, width, height, confidence, t_norm


def extract_rich_sequence(fragment: Dict) -> np.ndarray:
    """
    Extract 8-feature trajectory sequence from a fragment dictionary.

    Returns:
        Array of shape (seq_len, 8) with
        [x_position, y_position, velocity, length, width, height, confidence, t_norm]
    """
    timestamps = np.array(fragment['timestamp'], dtype=np.float64)
    seq_len = len(timestamps)
    t_norm = timestamps - timestamps[0]

    x_pos = np.array(fragment['x_position'], dtype=np.float32)
    y_pos = np.array(fragment['y_position'], dtype=np.float32)

    # Velocity
    if 'velocity' in fragment and fragment['velocity'] is not None:
        vel = np.array(fragment['velocity'], dtype=np.float32)
        if len(vel) != seq_len:
            vel = _compute_velocity(timestamps, x_pos)
    else:
        vel = _compute_velocity(timestamps, x_pos)

    # Length - scalar or array
    length = _to_array(fragment.get('length', 15.0), seq_len)
    width = _to_array(fragment.get('width', 6.0), seq_len)
    height = _to_array(fragment.get('height', 5.0), seq_len)

    # Detection confidence
    if 'detection_confidence' in fragment and fragment['detection_confidence'] is not None:
        conf = np.array(fragment['detection_confidence'], dtype=np.float32)
        if len(conf) != seq_len:
            conf = np.ones(seq_len, dtype=np.float32)
    else:
        conf = np.ones(seq_len, dtype=np.float32)

    sequence = np.column_stack([
        x_pos, y_pos, vel, length, width, height, conf,
        t_norm.astype(np.float32)
    ])
    return sequence.astype(np.float32)


def extract_endpoint_features(frag_a: Dict, frag_b: Dict) -> np.ndarray:
    """
    Extract 4 endpoint gap features between two fragments.

    Returns:
        Array of shape (4,) with [time_gap, x_gap, y_gap, velocity_diff]
    """
    t_a_end = frag_a.get('last_timestamp', frag_a['timestamp'][-1])
    t_b_start = frag_b.get('first_timestamp', frag_b['timestamp'][0])
    time_gap = t_b_start - t_a_end

    x_a_end = frag_a.get('ending_x', frag_a['x_position'][-1])
    x_b_start = frag_b.get('starting_x', frag_b['x_position'][0])
    x_gap = x_b_start - x_a_end

    y_gap = frag_b['y_position'][0] - frag_a['y_position'][-1]

    v_a = _get_endpoint_velocity(frag_a, end=True)
    v_b = _get_endpoint_velocity(frag_b, end=False)
    velocity_diff = v_b - v_a

    return np.array([time_gap, x_gap, y_gap, velocity_diff], dtype=np.float32)


def _compute_velocity(timestamps, x_pos):
    """Compute velocity from positions and timestamps."""
    dt = np.diff(timestamps)
    dx = np.diff(x_pos)
    velocity = np.zeros(len(timestamps), dtype=np.float32)
    velocity[1:] = dx / (dt + 1e-6)
    velocity[0] = velocity[1] if len(velocity) > 1 else 0
    return velocity


def _to_array(value, length):
    """Convert scalar or list to numpy array of given length."""
    if isinstance(value, (list, np.ndarray)):
        arr = np.array(value, dtype=np.float32)
        if len(arr) == length:
            return arr
        return np.full(length, np.mean(arr), dtype=np.float32)
    return np.full(length, float(value), dtype=np.float32)


def _get_endpoint_velocity(frag, end=True):
    """Get velocity at endpoint of fragment."""
    if 'velocity' in frag and frag['velocity'] is not None and len(frag['velocity']) > 0:
        return frag['velocity'][-1] if end else frag['velocity'][0]
    ts = frag['timestamp']
    xs = frag['x_position']
    if len(ts) >= 2:
        if end:
            return (xs[-1] - xs[-2]) / (ts[-1] - ts[-2] + 1e-6)
        else:
            return (xs[1] - xs[0]) / (ts[1] - ts[0] + 1e-6)
    return 0.0


def _get_gt_id(fragment: Dict) -> Optional[str]:
    """Extract ground truth vehicle ID from fragment."""
    if 'gt_ids' in fragment and len(fragment['gt_ids']) > 0:
        first_gt_id = fragment['gt_ids'][0]
        if isinstance(first_gt_id, list) and len(first_gt_id) > 0:
            if isinstance(first_gt_id[0], dict) and '$oid' in first_gt_id[0]:
                return first_gt_id[0]['$oid']
    if '_source_gt_id' in fragment:
        return fragment['_source_gt_id']
    return None


class RichSequenceDataset(Dataset):
    """
    Dataset yielding 8-feature raw sequences for each fragment pair.
    Used by TCN and Transformer models.

    Each sample: (seq_a, seq_b, endpoint_features, label)
    """

    def __init__(
        self,
        dataset_names: List[str] = ['i', 'ii', 'iii'],
        data_dir: str = None,
        max_time_gap: float = 5.0,
        normalize: bool = True,
        random_seed: int = 42,
        negative_sampling_mode: str = "stratified",
        negative_gap_bins: Tuple[float, float, float] = (5.0, 10.0, 15.0),
        negative_bin_weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
        negative_time_gap_max: float = 15.0,
    ):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent / "Siamese-Network" / "data")
        self.data_dir = Path(data_dir)
        self.dataset_names = dataset_names
        self.max_time_gap = max_time_gap
        self.normalize = normalize
        self.random_seed = int(random_seed)
        self._rng = random.Random(self.random_seed)
        self.negative_sampling_mode = str(negative_sampling_mode).lower()
        self.negative_gap_bins = tuple(float(v) for v in negative_gap_bins)
        self.negative_bin_weights = tuple(float(v) for v in negative_bin_weights)
        self.negative_time_gap_max = float(negative_time_gap_max)
        self._validate_negative_sampling_config()

        self.pairs: List[Dict] = []
        self.norm_stats = None

        self._generate_pairs()
        if self.normalize:
            self._compute_normalization_stats()

    def _validate_negative_sampling_config(self) -> None:
        if self.negative_sampling_mode not in {"stratified", "uniform"}:
            raise ValueError(
                f"negative_sampling_mode must be 'stratified' or 'uniform', "
                f"got '{self.negative_sampling_mode}'."
            )
        if len(self.negative_gap_bins) != len(self.negative_bin_weights):
            raise ValueError(
                f"negative_gap_bins ({len(self.negative_gap_bins)}) and "
                f"negative_bin_weights ({len(self.negative_bin_weights)}) must match."
            )
        if any(v <= 0 for v in self.negative_gap_bins):
            raise ValueError(f"negative_gap_bins must be positive, got {self.negative_gap_bins}.")
        if any(v <= 0 for v in self.negative_bin_weights):
            raise ValueError(
                f"negative_bin_weights must be positive, got {self.negative_bin_weights}."
            )
        if list(self.negative_gap_bins) != sorted(self.negative_gap_bins):
            raise ValueError(f"negative_gap_bins must be increasing, got {self.negative_gap_bins}.")
        if abs(sum(self.negative_bin_weights) - 1.0) > 1e-6:
            raise ValueError(
                f"negative_bin_weights must sum to 1.0, got {sum(self.negative_bin_weights):.6f}."
            )
        if abs(self.negative_gap_bins[-1] - self.negative_time_gap_max) > 1e-6:
            raise ValueError(
                f"Last negative_gap_bins edge ({self.negative_gap_bins[-1]}) must equal "
                f"negative_time_gap_max ({self.negative_time_gap_max})."
            )
        if self.negative_time_gap_max < self.max_time_gap:
            raise ValueError(
                f"negative_time_gap_max ({self.negative_time_gap_max}) must be >= "
                f"max_time_gap ({self.max_time_gap})."
            )

    def _gap_bin_index(self, gap: float) -> Optional[int]:
        """
        Map a positive gap to its configured bin index.
        Bins are (0, b0], (b0, b1], ..., (b_{k-2}, b_{k-1}].
        """
        if gap <= 0 or gap > self.negative_time_gap_max:
            return None
        idx = bisect_left(self.negative_gap_bins, gap)
        return min(max(idx, 0), len(self.negative_gap_bins) - 1)

    def _build_negative_candidate_bins(self, raw_data: List[Dict]) -> List[List[Dict]]:
        """
        Build candidate negatives bucketed by time-gap bins.
        """
        bins = [[] for _ in self.negative_gap_bins]
        by_direction: Dict[str, List[Tuple[float, float, Optional[str], Dict]]] = defaultdict(list)

        for frag in raw_data:
            gt_id = _get_gt_id(frag)
            start = float(frag.get('first_timestamp', frag['timestamp'][0]))
            end = float(frag.get('last_timestamp', frag['timestamp'][-1]))
            direction = str(frag.get('direction', 'unknown'))
            by_direction[direction].append((start, end, gt_id, frag))

        for direction, entries in by_direction.items():
            entries.sort(key=lambda x: x[0])  # sort by start time
            starts = [e[0] for e in entries]
            n = len(entries)
            for i in range(n):
                _start_a, end_a, ga, fa = entries[i]
                if ga is None:
                    continue

                lo = bisect_right(starts, end_a)
                hi = bisect_right(starts, end_a + self.negative_time_gap_max)
                if lo >= hi:
                    continue

                for j in range(lo, hi):
                    start_b, _end_b, gb, fb = entries[j]
                    if gb is None or ga == gb:
                        continue
                    gap = start_b - end_a
                    bin_idx = self._gap_bin_index(gap)
                    if bin_idx is None:
                        continue
                    bins[bin_idx].append({
                        "fa": fa,
                        "fb": fb,
                        "label": 0,
                        "ga": ga,
                        "gb": gb,
                        "gap": float(gap),
                        "direction": direction,
                    })
        return bins

    def _sample_negatives(self, bins: List[List[Dict]], target: int, dataset_name: str) -> List[Dict]:
        if target <= 0:
            return []

        if self.negative_sampling_mode == "uniform":
            pool = []
            for bucket in bins:
                pool.extend(bucket)
            self._rng.shuffle(pool)
            selected = pool[:target]
            for item in selected:
                item["dataset"] = dataset_name
            return selected

        # Stratified sampling
        raw_quotas = [target * w for w in self.negative_bin_weights]
        quotas = [int(np.floor(v)) for v in raw_quotas]
        remainder = target - sum(quotas)
        if remainder > 0:
            frac_order = sorted(
                range(len(raw_quotas)),
                key=lambda i: raw_quotas[i] - quotas[i],
                reverse=True,
            )
            for i in frac_order[:remainder]:
                quotas[i] += 1

        selected_bins: List[List[Dict]] = []
        unused_bins: List[List[Dict]] = []
        for i, bucket in enumerate(bins):
            bucket_copy = list(bucket)
            self._rng.shuffle(bucket_copy)
            take = min(quotas[i], len(bucket_copy))
            selected_bins.append(bucket_copy[:take])
            unused_bins.append(bucket_copy[take:])

        selected = []
        for part in selected_bins:
            selected.extend(part)
        shortfall = target - len(selected)

        if shortfall > 0:
            extra_pool = []
            for part in unused_bins:
                extra_pool.extend(part)
            self._rng.shuffle(extra_pool)
            selected.extend(extra_pool[:shortfall])

        # Annotate dataset name for downstream splits/debugging.
        for item in selected:
            item["dataset"] = dataset_name
        return selected

    def _load_dataset(self, name: str):
        gt_path = self.data_dir / f"GT_{name}.json"
        raw_path = self.data_dir / f"RAW_{name}.json"
        # Handle renamed files
        if not raw_path.exists():
            raw_bhat = self.data_dir / f"RAW_{name}_Bhat.json"
            if raw_bhat.exists():
                raw_path = raw_bhat

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        with open(raw_path, 'r') as f:
            raw_data = json.load(f)
        print(f"Loaded {name}: {len(gt_data)} GT, {len(raw_data)} RAW")
        return gt_data, raw_data

    def _generate_pairs(self):
        """Generate balanced positive/negative pairs from all datasets."""
        print("=" * 60)
        print("GENERATING RICH SEQUENCE DATASET (8 features)")
        print("=" * 60)

        all_positives = []
        all_negatives = []

        for name in self.dataset_names:
            print(f"\nProcessing: {name}")
            _, raw_data = self._load_dataset(name)

            # Positive pairs: same gt_id, sequential
            vehicle_frags = defaultdict(list)
            for frag in raw_data:
                gt_id = _get_gt_id(frag)
                if gt_id:
                    vehicle_frags[gt_id].append(frag)

            positives = []
            for gt_id, frags in vehicle_frags.items():
                frags_sorted = sorted(frags, key=lambda f: f.get('first_timestamp', f['timestamp'][0]))
                for i in range(len(frags_sorted) - 1):
                    fa, fb = frags_sorted[i], frags_sorted[i + 1]
                    t_end = fa.get('last_timestamp', fa['timestamp'][-1])
                    t_start = fb.get('first_timestamp', fb['timestamp'][0])
                    if t_end < t_start and (t_start - t_end) <= self.max_time_gap:
                        if fa.get('direction') == fb.get('direction'):
                            positives.append({
                                "fa": fa,
                                "fb": fb,
                                "label": 1,
                                "ga": gt_id,
                                "gb": gt_id,
                                "dataset": name,
                            })

            print(f"  Positives: {len(positives)}")
            all_positives.extend(positives)

            # Negative pairs: different gt_id, same direction, across full runtime window.
            target = len(positives)
            negative_bins = self._build_negative_candidate_bins(raw_data)
            bin_candidates = [len(b) for b in negative_bins]
            negatives = self._sample_negatives(negative_bins, target=target, dataset_name=name)

            if self.negative_sampling_mode == "stratified":
                selected_bins = [0 for _ in self.negative_gap_bins]
                for pair in negatives:
                    idx = self._gap_bin_index(float(pair.get("gap", -1)))
                    if idx is not None:
                        selected_bins[idx] += 1
                edges = [0.0] + list(self.negative_gap_bins)
                print("  Negative bin coverage:")
                for i in range(len(self.negative_gap_bins)):
                    print(
                        f"    ({edges[i]:.1f}, {edges[i+1]:.1f}]s: "
                        f"candidates={bin_candidates[i]}, selected={selected_bins[i]}"
                    )

            if len(negatives) < target:
                print(
                    f"  Warning: only sampled {len(negatives)} negatives for target={target} "
                    f"within <= {self.negative_time_gap_max:.1f}s."
                )

            print(f"  Negatives: {len(negatives)}")
            all_negatives.extend(negatives)

        self.pairs = all_positives + all_negatives
        print(f"\nTotal: {len(self.pairs)} ({len(all_positives)} pos, {len(all_negatives)} neg)")

    def _compute_normalization_stats(self):
        """Compute normalization stats from all pairs with balanced sampling."""
        print("Computing normalization stats...")
        indices = list(range(len(self.pairs)))
        self.compute_normalization_stats_from_indices(indices, sample_size=1000, balanced=True)

    def compute_normalization_stats_from_indices(
        self,
        indices: List[int],
        sample_size: int = 1000,
        balanced: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute sequence + endpoint normalization stats from selected pair indices.
        """
        if indices is None or len(indices) == 0:
            raise ValueError("indices must be non-empty for normalization stats.")

        candidate = np.asarray(indices, dtype=np.int64)
        rng = np.random.default_rng(self.random_seed)

        if balanced:
            pos_idx = candidate[np.array([self.pairs[int(i)]["label"] == 1 for i in candidate])]
            neg_idx = candidate[np.array([self.pairs[int(i)]["label"] == 0 for i in candidate])]

            selected = []
            half = max(1, sample_size // 2)
            if len(pos_idx) > 0:
                n_pos = min(half, len(pos_idx))
                selected.extend(rng.choice(pos_idx, size=n_pos, replace=False).tolist())
            if len(neg_idx) > 0:
                n_neg = min(half, len(neg_idx))
                selected.extend(rng.choice(neg_idx, size=n_neg, replace=False).tolist())

            selected_set = set(int(i) for i in selected)
            remaining_budget = max(0, min(sample_size, len(candidate)) - len(selected))
            if remaining_budget > 0:
                remaining = [int(i) for i in candidate if int(i) not in selected_set]
                if remaining:
                    n_extra = min(remaining_budget, len(remaining))
                    selected.extend(rng.choice(np.asarray(remaining), size=n_extra, replace=False).tolist())
        else:
            n = min(sample_size, len(candidate))
            selected = rng.choice(candidate, size=n, replace=False).tolist()

        if not selected:
            raise RuntimeError("Failed to sample pairs for normalization stats.")

        seq_rows = []
        ep_rows = []
        for idx in selected:
            pair = self.pairs[int(idx)]
            fa = pair["fa"]
            fb = pair["fb"]
            seq_rows.append(extract_rich_sequence(fa))
            seq_rows.append(extract_rich_sequence(fb))
            ep_rows.append(extract_endpoint_features(fa, fb))

        seq_data = np.vstack(seq_rows)
        ep_data = np.vstack(ep_rows)
        stats = {
            "mean": seq_data.mean(axis=0).astype(np.float32),
            "std": (seq_data.std(axis=0) + 1e-6).astype(np.float32),
            "ep_mean": ep_data.mean(axis=0).astype(np.float32),
            "ep_std": (ep_data.std(axis=0) + 1e-6).astype(np.float32),
        }
        self.norm_stats = stats
        return stats

    def set_normalization_stats(self, stats: Dict[str, np.ndarray]):
        if stats is None:
            raise ValueError("stats cannot be None")
        required = ("mean", "std", "ep_mean", "ep_std")
        missing = [k for k in required if k not in stats]
        if missing:
            raise ValueError(f"Missing normalization keys: {missing}")
        self.norm_stats = {
            "mean": np.asarray(stats["mean"], dtype=np.float32),
            "std": np.asarray(stats["std"], dtype=np.float32),
            "ep_mean": np.asarray(stats["ep_mean"], dtype=np.float32),
            "ep_std": np.asarray(stats["ep_std"], dtype=np.float32),
        }

    def _normalize(self, seq: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return seq
        return (seq - self.norm_stats['mean']) / self.norm_stats['std']

    def _normalize_endpoint(self, ep: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return ep
        return (ep - self.norm_stats["ep_mean"]) / self.norm_stats["ep_std"]

    def get_pair_vehicle_ids(self, idx: int) -> Tuple[Optional[str], Optional[str]]:
        pair = self.pairs[idx]
        return pair.get("ga"), pair.get("gb")

    def get_pair_label(self, idx: int) -> int:
        return int(self.pairs[idx]["label"])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        fa, fb, label = pair["fa"], pair["fb"], pair["label"]
        seq_a = extract_rich_sequence(fa)
        seq_b = extract_rich_sequence(fb)
        ep = extract_endpoint_features(fa, fb)

        if self.normalize:
            seq_a = self._normalize(seq_a)
            seq_b = self._normalize(seq_b)
            ep = self._normalize_endpoint(ep)

        return (
            torch.FloatTensor(seq_a),
            torch.FloatTensor(seq_b),
            torch.FloatTensor(ep),
            torch.FloatTensor([label]),
        )


def rich_collate_fn(batch):
    """Collate variable-length sequences with padding."""
    seqs_a, seqs_b, eps, labels = zip(*batch)

    max_a = max(s.size(0) for s in seqs_a)
    max_b = max(s.size(0) for s in seqs_b)
    nf = seqs_a[0].size(1)

    padded_a = torch.zeros(len(seqs_a), max_a, nf)
    padded_b = torch.zeros(len(seqs_b), max_b, nf)
    len_a = torch.LongTensor([s.size(0) for s in seqs_a])
    len_b = torch.LongTensor([s.size(0) for s in seqs_b])

    # Padding masks (True = padded position)
    mask_a = torch.ones(len(seqs_a), max_a, dtype=torch.bool)
    mask_b = torch.ones(len(seqs_b), max_b, dtype=torch.bool)

    for i, (sa, sb) in enumerate(zip(seqs_a, seqs_b)):
        padded_a[i, :sa.size(0)] = sa
        padded_b[i, :sb.size(0)] = sb
        mask_a[i, :sa.size(0)] = False
        mask_b[i, :sb.size(0)] = False

    return padded_a, len_a, mask_a, padded_b, len_b, mask_b, torch.stack(eps), torch.stack(labels)


if __name__ == "__main__":
    ds = RichSequenceDataset(dataset_names=['i'], normalize=True)
    print(f"Dataset size: {len(ds)}")
    sa, sb, ep, lbl = ds[0]
    print(f"Seq A: {sa.shape}, Seq B: {sb.shape}, Endpoint: {ep.shape}, Label: {lbl.item()}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=rich_collate_fn)
    batch = next(iter(loader))
    pa, la, ma, pb, lb, mb, eps, labels = batch
    print(f"Batch: A={pa.shape}, B={pb.shape}, mask_A={ma.shape}, eps={eps.shape}")
