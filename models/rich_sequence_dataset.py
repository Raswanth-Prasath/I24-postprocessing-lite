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
    ):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent / "Siamese-Network" / "data")
        self.data_dir = Path(data_dir)
        self.dataset_names = dataset_names
        self.max_time_gap = max_time_gap
        self.normalize = normalize

        self.pairs: List[Tuple[Dict, Dict, int]] = []
        self.norm_stats = None

        self._generate_pairs()
        if self.normalize:
            self._compute_normalization_stats()

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
                            positives.append((fa, fb, 1))

            print(f"  Positives: {len(positives)}")
            all_positives.extend(positives)

            # Negative pairs: different gt_id, same direction, sequential
            import random
            target = len(positives)
            negatives = []
            attempts = 0
            while len(negatives) < target and attempts < target * 50:
                attempts += 1
                ia = random.randint(0, len(raw_data) - 1)
                ib = random.randint(0, len(raw_data) - 1)
                if ia == ib:
                    continue
                fa, fb = raw_data[ia], raw_data[ib]
                ga, gb = _get_gt_id(fa), _get_gt_id(fb)
                if ga is None or gb is None or ga == gb:
                    continue
                if fa.get('direction') != fb.get('direction'):
                    continue
                t_end = fa.get('last_timestamp', fa['timestamp'][-1])
                t_start = fb.get('first_timestamp', fb['timestamp'][0])
                if t_end < t_start and (t_start - t_end) <= self.max_time_gap * 2:
                    negatives.append((fa, fb, 0))

            print(f"  Negatives: {len(negatives)}")
            all_negatives.extend(negatives)

        self.pairs = all_positives + all_negatives
        print(f"\nTotal: {len(self.pairs)} ({len(all_positives)} pos, {len(all_negatives)} neg)")

    def _compute_normalization_stats(self):
        """Compute per-feature mean/std from a sample of sequences."""
        print("Computing normalization stats...")
        sample_size = min(1000, len(self.pairs))
        all_seqs = []
        for fa, fb, _ in self.pairs[:sample_size]:
            all_seqs.append(extract_rich_sequence(fa))
            all_seqs.append(extract_rich_sequence(fb))
        data = np.vstack(all_seqs)
        self.norm_stats = {
            'mean': data.mean(axis=0).astype(np.float32),
            'std': (data.std(axis=0) + 1e-6).astype(np.float32),
        }

    def _normalize(self, seq: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return seq
        return (seq - self.norm_stats['mean']) / self.norm_stats['std']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fa, fb, label = self.pairs[idx]
        seq_a = extract_rich_sequence(fa)
        seq_b = extract_rich_sequence(fb)
        ep = extract_endpoint_features(fa, fb)

        if self.normalize:
            seq_a = self._normalize(seq_a)
            seq_b = self._normalize(seq_b)

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
