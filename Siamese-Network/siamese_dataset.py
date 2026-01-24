"""
Siamese Network Dataset Loader for Trajectory Fragment Pairs

This module creates paired trajectory sequences (not hand-crafted features)
for training the Siamese network to learn similarity metrics.

Enhanced version includes:
- Trajectory masking for synthetic positive pair generation
- Hard negative mining for challenging negative pairs
- Endpoint feature extraction for improved similarity head
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Import enhanced data pipeline components
try:
    from trajectory_masking import TrajectoryMasker, KinematicAugmenter, MaskConfig
    from hard_negative_mining import HardNegativeMiner, HardNegativeConfig
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError:
    ENHANCED_PIPELINE_AVAILABLE = False
    print("Warning: Enhanced pipeline modules not found. Using basic dataset.")


class TrajectoryPairDataset(Dataset):
    """
    Dataset for Siamese Network training on trajectory fragment pairs.

    Unlike the logistic regression approach that uses engineered features,
    this dataset provides raw trajectory sequences for the LSTM encoder to learn from.
    """

    def __init__(self,
                 dataset_names: List[str] = ['i', 'ii', 'iii'],
                 max_time_gap: float = 5.0,
                 max_spatial_gap: float = 200,
                 y_threshold: float = 5.0,
                 sequence_features: List[str] = ['x_position', 'y_position', 'velocity'],
                 normalize: bool = True):
        """
        Args:
            dataset_names: Which I24-3D scenarios to use
            max_time_gap: Maximum time gap between fragments (seconds)
            max_spatial_gap: Maximum spatial gap (feet)
            y_threshold: Maximum lateral distance for same lane (feet)
            sequence_features: Which features to include in sequences
            normalize: Whether to normalize sequences
        """
        # Use relative path to data folder in same directory
        # This makes the folder self-contained and portable to Sol
        script_dir = Path(__file__).parent
        self.dataset_dir = script_dir / "data"
        self.dataset_names = dataset_names
        self.max_time_gap = max_time_gap
        self.max_spatial_gap = max_spatial_gap
        self.y_threshold = y_threshold
        self.sequence_features = sequence_features
        self.normalize = normalize

        # Storage for pairs
        self.pairs = []
        self.labels = []

        # Normalization statistics
        self.norm_stats = None

        # Generate dataset
        self._generate_pairs()

        if self.normalize:
            self._compute_normalization_stats()

    def _load_dataset(self, dataset_name: str) -> Tuple[List[Dict], List[Dict]]:
        """Load GT and RAW data for a specific dataset"""
        gt_path = self.dataset_dir / f"GT_{dataset_name}.json"
        raw_path = self.dataset_dir / f"RAW_{dataset_name}.json"

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        with open(raw_path, 'r') as f:
            raw_data = json.load(f)

        print(f"Loaded {dataset_name}: {len(gt_data)} GT trajectories, {len(raw_data)} RAW fragments")
        return gt_data, raw_data

    def _get_gt_id(self, fragment: Dict) -> str:
        """Extract ground truth vehicle ID from fragment"""
        if 'gt_ids' in fragment and len(fragment['gt_ids']) > 0:
            return fragment['gt_ids'][0][0]['$oid']
        return None

    def _is_sequential(self, frag_a: Dict, frag_b: Dict) -> bool:
        """Check if frag_b comes after frag_a without overlap"""
        return frag_a['last_timestamp'] < frag_b['first_timestamp']

    def _is_candidate_pair(self, frag_a: Dict, frag_b: Dict) -> bool:
        """Check if two fragments could potentially be stitched"""
        # Must be sequential
        if not self._is_sequential(frag_a, frag_b):
            return False

        # Must have same direction
        if frag_a['direction'] != frag_b['direction']:
            return False

        # Time gap constraint
        time_gap = frag_b['first_timestamp'] - frag_a['last_timestamp']
        if time_gap > self.max_time_gap:
            return False

        # Spatial gap constraint
        if frag_a['direction'] == 1:  # Eastbound
            spatial_gap = frag_b['starting_x'] - frag_a['ending_x']
        else:  # Westbound
            spatial_gap = frag_a['starting_x'] - frag_b['ending_x']

        if spatial_gap < -50 or spatial_gap > self.max_spatial_gap:
            return False

        # Y-position constraint
        y_diff = abs(np.mean(frag_b['y_position']) - np.mean(frag_a['y_position']))
        if y_diff > self.y_threshold:
            return False

        return True

    def _extract_sequence(self, fragment: Dict) -> np.ndarray:
        """
        Extract raw trajectory sequence from fragment

        Returns:
            Array of shape (seq_len, num_features)
            Features: [x, y, velocity, timestamp_normalized]
        """
        seq_data = []

        # Get timestamps
        timestamps = np.array(fragment['timestamp'])
        t_start = timestamps[0]
        t_norm = timestamps - t_start  # Normalize to start at 0

        # Get positions
        x_pos = np.array(fragment['x_position'])
        y_pos = np.array(fragment['y_position'])

        # Get velocity
        if 'velocity' in fragment and len(fragment['velocity']) > 0:
            velocity = np.array(fragment['velocity'])
        else:
            # Compute velocity from positions if not available
            dt = np.diff(timestamps)
            dx = np.diff(x_pos)
            velocity = np.zeros(len(timestamps))
            velocity[1:] = dx / (dt + 1e-6)
            velocity[0] = velocity[1]  # Use next velocity for first point

        # Build sequence
        for i in range(len(timestamps)):
            point = [x_pos[i], y_pos[i], velocity[i], t_norm[i]]
            seq_data.append(point)

        return np.array(seq_data, dtype=np.float32)

    def _generate_positive_pairs(self, raw_fragments: List[Dict]) -> List[Tuple[int, int]]:
        """Generate positive pairs (same vehicle)"""
        positive_pairs = []

        # Group fragments by GT vehicle ID
        vehicle_fragments = defaultdict(list)
        for idx, frag in enumerate(raw_fragments):
            gt_id = self._get_gt_id(frag)
            if gt_id is not None:
                vehicle_fragments[gt_id].append(idx)

        # Generate pairs within each vehicle
        for gt_id, fragment_indices in vehicle_fragments.items():
            for i, idx_a in enumerate(fragment_indices):
                for idx_b in fragment_indices[i+1:]:
                    frag_a = raw_fragments[idx_a]
                    frag_b = raw_fragments[idx_b]

                    if self._is_candidate_pair(frag_a, frag_b):
                        positive_pairs.append((idx_a, idx_b))

        return positive_pairs

    def _generate_hard_negatives(self, raw_fragments: List[Dict],
                                  num_negatives: int) -> List[Tuple[int, int]]:
        """Generate hard negative pairs (different vehicles but spatially/temporally close)"""
        negative_pairs = []

        # Build spatial-temporal index
        time_window = 2.0  # seconds
        space_window = 150  # feet

        st_index = defaultdict(list)
        for idx, frag in enumerate(raw_fragments):
            t_bin = int(frag['first_timestamp'] / time_window)
            x_bin = int(frag['starting_x'] / space_window)
            direction = frag['direction']
            st_index[(t_bin, x_bin, direction)].append(idx)

        # Sample negative pairs
        attempted = 0
        max_attempts = num_negatives * 20
        search_radius = 2

        for (t_bin, x_bin, direction), fragment_indices in st_index.items():
            if len(negative_pairs) >= num_negatives:
                break

            # Look in neighboring bins
            for dt in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    neighbor_key = (t_bin + dt, x_bin + dx, direction)
                    if neighbor_key not in st_index:
                        continue

                    neighbor_indices = st_index[neighbor_key]

                    for idx_a in fragment_indices:
                        for idx_b in neighbor_indices:
                            if idx_a >= idx_b:
                                continue

                            attempted += 1
                            if attempted >= max_attempts:
                                break

                            frag_a = raw_fragments[idx_a]
                            frag_b = raw_fragments[idx_b]

                            # Check if from different vehicles
                            gt_a = self._get_gt_id(frag_a)
                            gt_b = self._get_gt_id(frag_b)

                            if gt_a is None or gt_b is None or gt_a == gt_b:
                                continue

                            # Check basic sequential and direction
                            if not self._is_sequential(frag_a, frag_b):
                                continue
                            if frag_a['direction'] != frag_b['direction']:
                                continue

                            # Relaxed time constraint
                            time_gap = frag_b['first_timestamp'] - frag_a['last_timestamp']
                            if time_gap > self.max_time_gap * 2:
                                continue

                            negative_pairs.append((idx_a, idx_b))

                            if len(negative_pairs) >= num_negatives:
                                return negative_pairs

        return negative_pairs

    def _generate_pairs(self):
        """Generate all positive and negative pairs from all datasets"""
        print("\n" + "="*60)
        print("GENERATING SIAMESE NETWORK TRAINING PAIRS")
        print("="*60)

        all_positives = []
        all_negatives = []

        for dataset_name in self.dataset_names:
            print(f"\nProcessing dataset: {dataset_name}")
            gt_data, raw_data = self._load_dataset(dataset_name)

            # Generate positive pairs
            print("Generating positive pairs...")
            positives = self._generate_positive_pairs(raw_data)
            print(f"Generated {len(positives)} positive pairs")

            # Store pairs with raw data reference
            for idx_a, idx_b in positives:
                all_positives.append((raw_data[idx_a], raw_data[idx_b], 1))

            # Generate hard negative pairs
            print("Mining hard negative pairs...")
            negatives = self._generate_hard_negatives(raw_data, num_negatives=len(positives))
            print(f"Generated {len(negatives)} negative pairs")

            # Store pairs
            for idx_a, idx_b in negatives:
                all_negatives.append((raw_data[idx_a], raw_data[idx_b], 0))

        # Combine all pairs
        self.pairs = all_positives + all_negatives

        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total pairs: {len(self.pairs)}")
        print(f"Positive pairs: {len(all_positives)} ({len(all_positives)/len(self.pairs)*100:.1f}%)")
        print(f"Negative pairs: {len(all_negatives)} ({len(all_negatives)/len(self.pairs)*100:.1f}%)")

    def _compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        print("\nComputing normalization statistics...")

        all_sequences = []
        for frag_a, frag_b, _ in tqdm(self.pairs[:1000]):  # Sample for efficiency
            seq_a = self._extract_sequence(frag_a)
            seq_b = self._extract_sequence(frag_b)
            all_sequences.append(seq_a)
            all_sequences.append(seq_b)

        # Concatenate all sequences
        all_data = np.vstack(all_sequences)

        self.norm_stats = {
            'mean': np.mean(all_data, axis=0),
            'std': np.std(all_data, axis=0) + 1e-6
        }

        print(f"Normalization stats computed:")
        print(f"  Mean: {self.norm_stats['mean']}")
        print(f"  Std: {self.norm_stats['std']}")

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize sequence using computed statistics"""
        if self.norm_stats is None:
            return sequence
        return (sequence - self.norm_stats['mean']) / self.norm_stats['std']

    def _extract_endpoint_features(self, frag_a: Dict, frag_b: Dict) -> np.ndarray:
        """
        Extract endpoint features for the similarity head.

        These 4 features capture the gap between fragments:
        - time_gap: temporal gap in seconds
        - x_gap: x-position gap in feet
        - y_gap: y-position gap in feet
        - velocity_diff: velocity difference in ft/s

        Args:
            frag_a: First fragment (anchor)
            frag_b: Second fragment (candidate)

        Returns:
            Array of shape (4,) with endpoint features
        """
        # Time gap
        t_a_end = frag_a.get('last_timestamp', frag_a['timestamp'][-1])
        t_b_start = frag_b.get('first_timestamp', frag_b['timestamp'][0])
        time_gap = t_b_start - t_a_end

        # X-position gap
        x_a_end = frag_a.get('ending_x', frag_a['x_position'][-1])
        x_b_start = frag_b.get('starting_x', frag_b['x_position'][0])
        x_gap = x_b_start - x_a_end

        # Y-position gap
        y_a_end = frag_a['y_position'][-1]
        y_b_start = frag_b['y_position'][0]
        y_gap = y_b_start - y_a_end

        # Velocity difference
        if 'velocity' in frag_a and len(frag_a['velocity']) > 0:
            v_a_end = frag_a['velocity'][-1]
        else:
            # Compute from positions
            if len(frag_a['timestamp']) >= 2:
                dt = frag_a['timestamp'][-1] - frag_a['timestamp'][-2]
                dx = frag_a['x_position'][-1] - frag_a['x_position'][-2]
                v_a_end = dx / (dt + 1e-6)
            else:
                v_a_end = 0

        if 'velocity' in frag_b and len(frag_b['velocity']) > 0:
            v_b_start = frag_b['velocity'][0]
        else:
            # Compute from positions
            if len(frag_b['timestamp']) >= 2:
                dt = frag_b['timestamp'][1] - frag_b['timestamp'][0]
                dx = frag_b['x_position'][1] - frag_b['x_position'][0]
                v_b_start = dx / (dt + 1e-6)
            else:
                v_b_start = 0

        velocity_diff = v_b_start - v_a_end

        return np.array([time_gap, x_gap, y_gap, velocity_diff], dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            seq_a: Tensor of shape (seq_len_a, num_features)
            seq_b: Tensor of shape (seq_len_b, num_features)
            endpoint_features: Tensor of shape (4,) with gap features
            label: 1 if same vehicle, 0 if different
        """
        frag_a, frag_b, label = self.pairs[idx]

        # Extract sequences
        seq_a = self._extract_sequence(frag_a)
        seq_b = self._extract_sequence(frag_b)

        # Extract endpoint features (4 features: time_gap, x_gap, y_gap, velocity_diff)
        endpoint_features = self._extract_endpoint_features(frag_a, frag_b)

        # Normalize sequences
        if self.normalize:
            seq_a = self._normalize_sequence(seq_a)
            seq_b = self._normalize_sequence(seq_b)

        # Convert to tensors
        seq_a = torch.FloatTensor(seq_a)
        seq_b = torch.FloatTensor(seq_b)
        endpoint_features = torch.FloatTensor(endpoint_features)
        label = torch.FloatTensor([label])

        return seq_a, seq_b, endpoint_features, label


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences

    Pads sequences to the maximum length in the batch.
    Also handles endpoint features (4-dim vector per pair).
    """
    seqs_a, seqs_b, endpoint_feats, labels = zip(*batch)

    # Get max lengths
    max_len_a = max(seq.size(0) for seq in seqs_a)
    max_len_b = max(seq.size(0) for seq in seqs_b)
    num_features = seqs_a[0].size(1)

    # Pad sequences
    padded_a = torch.zeros(len(seqs_a), max_len_a, num_features)
    padded_b = torch.zeros(len(seqs_b), max_len_b, num_features)
    lengths_a = torch.LongTensor([seq.size(0) for seq in seqs_a])
    lengths_b = torch.LongTensor([seq.size(0) for seq in seqs_b])

    for i, (seq_a, seq_b) in enumerate(zip(seqs_a, seqs_b)):
        padded_a[i, :seq_a.size(0), :] = seq_a
        padded_b[i, :seq_b.size(0), :] = seq_b

    # Stack endpoint features and labels
    endpoint_features = torch.stack(endpoint_feats)
    labels = torch.stack(labels)

    return padded_a, lengths_a, padded_b, lengths_b, endpoint_features, labels


class EnhancedTrajectoryPairDataset(Dataset):
    """
    Enhanced dataset with trajectory masking and hard negative mining.

    Key improvements over TrajectoryPairDataset:
    - Generates ~10x more positive pairs via trajectory masking from GT data
    - Mines hard negatives with finer spatial-temporal binning (0.5s, 50ft)
    - Optional data augmentation (velocity noise, y-position noise, point dropout)
    - Returns endpoint features for improved similarity head
    """

    def __init__(self,
                 dataset_names: List[str] = ['i', 'ii', 'iii'],
                 max_time_gap: float = 5.0,
                 sequence_features: List[str] = ['x_position', 'y_position', 'velocity'],
                 normalize: bool = True,
                 use_masking: bool = True,
                 use_hard_negatives: bool = True,
                 augment: bool = False,
                 pairs_per_trajectory: int = 5,
                 hard_negative_ratio: float = 0.7):
        """
        Args:
            dataset_names: Which I24-3D scenarios to use
            max_time_gap: Maximum time gap between fragments (seconds)
            sequence_features: Which features to include in sequences
            normalize: Whether to normalize sequences
            use_masking: Generate positive pairs from GT via masking
            use_hard_negatives: Mine hard negatives (vs random negatives)
            augment: Apply data augmentation during training
            pairs_per_trajectory: Number of random mask pairs per GT trajectory
            hard_negative_ratio: Fraction of negatives that should be "hard"
        """
        if not ENHANCED_PIPELINE_AVAILABLE:
            raise ImportError(
                "Enhanced pipeline modules not available. "
                "Please ensure trajectory_masking.py and hard_negative_mining.py are in the same directory."
            )

        script_dir = Path(__file__).parent
        self.dataset_dir = script_dir / "data"
        self.dataset_names = dataset_names
        self.max_time_gap = max_time_gap
        self.sequence_features = sequence_features
        self.normalize = normalize
        self.use_masking = use_masking
        self.use_hard_negatives = use_hard_negatives
        self.augment = augment
        self.pairs_per_trajectory = pairs_per_trajectory
        self.hard_negative_ratio = hard_negative_ratio

        # Storage for pairs
        self.pairs = []

        # Normalization statistics
        self.norm_stats = None

        # Initialize masker and miner
        self.masker = TrajectoryMasker() if use_masking else None
        self.miner = HardNegativeMiner() if use_hard_negatives else None
        self.augmenter = KinematicAugmenter() if augment else None

        # Generate dataset
        self._generate_enhanced_pairs()

        if self.normalize:
            self._compute_normalization_stats()

    def _load_dataset(self, dataset_name: str) -> Tuple[List[Dict], List[Dict]]:
        """Load GT and RAW data for a specific dataset"""
        gt_path = self.dataset_dir / f"GT_{dataset_name}.json"
        raw_path = self.dataset_dir / f"RAW_{dataset_name}.json"

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        with open(raw_path, 'r') as f:
            raw_data = json.load(f)

        print(f"Loaded {dataset_name}: {len(gt_data)} GT trajectories, {len(raw_data)} RAW fragments")
        return gt_data, raw_data

    def _generate_enhanced_pairs(self):
        """Generate pairs using masking and hard negative mining"""
        print("\n" + "="*60)
        print("GENERATING ENHANCED SIAMESE NETWORK TRAINING PAIRS")
        print("="*60)

        all_positives = []
        all_negatives = []

        for dataset_name in self.dataset_names:
            print(f"\nProcessing dataset: {dataset_name}")
            gt_data, raw_data = self._load_dataset(dataset_name)

            # Generate positive pairs via masking (from GT)
            if self.use_masking and self.masker is not None:
                print("Generating positive pairs via trajectory masking...")
                masked_positives = self.masker.generate_all_pairs(
                    gt_data,
                    pairs_per_trajectory=self.pairs_per_trajectory,
                    include_random_masks=True
                )
                print(f"  Generated {len(masked_positives)} positive pairs from masking")
                all_positives.extend(masked_positives)
            else:
                # Fall back to RAW-based positive pairs
                print("Generating positive pairs from RAW fragments...")
                positives = self._generate_positive_pairs_from_raw(raw_data)
                print(f"  Generated {len(positives)} positive pairs")
                all_positives.extend(positives)

            # Target: balanced dataset (equal positives and negatives)
            target_negatives = len(all_positives) // len(self.dataset_names)

            # Mine hard negatives
            neg_pairs = []
            if self.use_hard_negatives and self.miner is not None:
                print("Mining hard negative pairs...")
                self.miner.build_index(raw_data)
                hard_negs = self.miner.mine_negatives_balanced(
                    num_negatives=target_negatives,
                    hard_ratio=self.hard_negative_ratio
                )
                neg_pairs = self.miner.get_fragment_pairs(hard_negs)
                print(f"  Mined {len(neg_pairs)} hard negative pairs")

            # Supplement with random negatives if hard negatives are insufficient
            if len(neg_pairs) < target_negatives:
                needed = target_negatives - len(neg_pairs)
                print(f"  Supplementing with {needed} random negative pairs...")
                random_negs = self._generate_random_negatives(raw_data, needed)
                neg_pairs.extend(random_negs)
                print(f"  Total negative pairs: {len(neg_pairs)}")

            all_negatives.extend(neg_pairs)

        # Combine all pairs
        self.pairs = all_positives + all_negatives

        print("\n" + "="*60)
        print("ENHANCED DATASET STATISTICS")
        print("="*60)
        print(f"Total pairs: {len(self.pairs)}")
        print(f"Positive pairs: {len(all_positives)} ({len(all_positives)/len(self.pairs)*100:.1f}%)")
        print(f"Negative pairs: {len(all_negatives)} ({len(all_negatives)/len(self.pairs)*100:.1f}%)")

    def _generate_positive_pairs_from_raw(self, raw_fragments: List[Dict]) -> List[Tuple[Dict, Dict, int]]:
        """Fall back method: generate positives from RAW using gt_ids"""
        positive_pairs = []
        vehicle_fragments = defaultdict(list)

        for idx, frag in enumerate(raw_fragments):
            gt_id = self._get_gt_id(frag)
            if gt_id is not None:
                vehicle_fragments[gt_id].append(frag)

        for gt_id, frags in vehicle_fragments.items():
            # Sort by time
            frags_sorted = sorted(frags, key=lambda f: f.get('first_timestamp', f['timestamp'][0]))
            for i in range(len(frags_sorted) - 1):
                frag_a = frags_sorted[i]
                frag_b = frags_sorted[i + 1]

                # Check if sequential
                t_a_end = frag_a.get('last_timestamp', frag_a['timestamp'][-1])
                t_b_start = frag_b.get('first_timestamp', frag_b['timestamp'][0])

                if t_a_end < t_b_start and (t_b_start - t_a_end) <= self.max_time_gap:
                    positive_pairs.append((frag_a, frag_b, 1))

        return positive_pairs

    def _generate_random_negatives(self, raw_fragments: List[Dict], num_negatives: int) -> List[Tuple[Dict, Dict, int]]:
        """Generate random negative pairs (different vehicles)"""
        import random
        negatives = []
        attempts = 0
        max_attempts = num_negatives * 50

        while len(negatives) < num_negatives and attempts < max_attempts:
            attempts += 1
            idx_a = random.randint(0, len(raw_fragments) - 1)
            idx_b = random.randint(0, len(raw_fragments) - 1)

            if idx_a == idx_b:
                continue

            frag_a = raw_fragments[idx_a]
            frag_b = raw_fragments[idx_b]

            gt_a = self._get_gt_id(frag_a)
            gt_b = self._get_gt_id(frag_b)

            # Different vehicles, same direction, sequential
            if gt_a is None or gt_b is None or gt_a == gt_b:
                continue
            if frag_a.get('direction') != frag_b.get('direction'):
                continue

            t_a_end = frag_a.get('last_timestamp', frag_a['timestamp'][-1])
            t_b_start = frag_b.get('first_timestamp', frag_b['timestamp'][0])

            if t_a_end < t_b_start and (t_b_start - t_a_end) <= self.max_time_gap * 2:
                negatives.append((frag_a, frag_b, 0))

        return negatives

    def _get_gt_id(self, fragment: Dict) -> Optional[str]:
        """Extract ground truth vehicle ID from fragment"""
        if 'gt_ids' in fragment and len(fragment['gt_ids']) > 0:
            first_gt_id = fragment['gt_ids'][0]
            if isinstance(first_gt_id, list) and len(first_gt_id) > 0:
                if isinstance(first_gt_id[0], dict) and '$oid' in first_gt_id[0]:
                    return first_gt_id[0]['$oid']
        if '_source_gt_id' in fragment:
            return fragment['_source_gt_id']
        return None

    def _extract_sequence(self, fragment: Dict) -> np.ndarray:
        """Extract raw trajectory sequence from fragment"""
        timestamps = np.array(fragment['timestamp'])
        t_start = timestamps[0]
        t_norm = timestamps - t_start

        x_pos = np.array(fragment['x_position'])
        y_pos = np.array(fragment['y_position'])

        if 'velocity' in fragment and len(fragment['velocity']) > 0:
            velocity = np.array(fragment['velocity'])
        else:
            dt = np.diff(timestamps)
            dx = np.diff(x_pos)
            velocity = np.zeros(len(timestamps))
            velocity[1:] = dx / (dt + 1e-6)
            velocity[0] = velocity[1] if len(velocity) > 1 else 0

        seq_data = []
        for i in range(len(timestamps)):
            point = [x_pos[i], y_pos[i], velocity[i], t_norm[i]]
            seq_data.append(point)

        return np.array(seq_data, dtype=np.float32)

    def _extract_endpoint_features(self, frag_a: Dict, frag_b: Dict) -> np.ndarray:
        """Extract endpoint features for the similarity head."""
        t_a_end = frag_a.get('last_timestamp', frag_a['timestamp'][-1])
        t_b_start = frag_b.get('first_timestamp', frag_b['timestamp'][0])
        time_gap = t_b_start - t_a_end

        x_a_end = frag_a.get('ending_x', frag_a['x_position'][-1])
        x_b_start = frag_b.get('starting_x', frag_b['x_position'][0])
        x_gap = x_b_start - x_a_end

        y_gap = frag_b['y_position'][0] - frag_a['y_position'][-1]

        if 'velocity' in frag_a and len(frag_a['velocity']) > 0:
            v_a_end = frag_a['velocity'][-1]
        else:
            if len(frag_a['timestamp']) >= 2:
                dt = frag_a['timestamp'][-1] - frag_a['timestamp'][-2]
                dx = frag_a['x_position'][-1] - frag_a['x_position'][-2]
                v_a_end = dx / (dt + 1e-6)
            else:
                v_a_end = 0

        if 'velocity' in frag_b and len(frag_b['velocity']) > 0:
            v_b_start = frag_b['velocity'][0]
        else:
            if len(frag_b['timestamp']) >= 2:
                dt = frag_b['timestamp'][1] - frag_b['timestamp'][0]
                dx = frag_b['x_position'][1] - frag_b['x_position'][0]
                v_b_start = dx / (dt + 1e-6)
            else:
                v_b_start = 0

        velocity_diff = v_b_start - v_a_end

        return np.array([time_gap, x_gap, y_gap, velocity_diff], dtype=np.float32)

    def _compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        print("\nComputing normalization statistics...")

        all_sequences = []
        sample_size = min(1000, len(self.pairs))
        for frag_a, frag_b, _ in tqdm(self.pairs[:sample_size]):
            seq_a = self._extract_sequence(frag_a)
            seq_b = self._extract_sequence(frag_b)
            all_sequences.append(seq_a)
            all_sequences.append(seq_b)

        all_data = np.vstack(all_sequences)

        self.norm_stats = {
            'mean': np.mean(all_data, axis=0),
            'std': np.std(all_data, axis=0) + 1e-6
        }

        print(f"Normalization stats computed from {sample_size} pairs")

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize sequence using computed statistics"""
        if self.norm_stats is None:
            return sequence
        return (sequence - self.norm_stats['mean']) / self.norm_stats['std']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            seq_a: Tensor of shape (seq_len_a, num_features)
            seq_b: Tensor of shape (seq_len_b, num_features)
            endpoint_features: Tensor of shape (4,) with gap features
            label: 1 if same vehicle, 0 if different
        """
        frag_a, frag_b, label = self.pairs[idx]

        # Apply augmentation if enabled
        if self.augment and self.augmenter is not None:
            if np.random.random() < 0.5:
                frag_a = self.augmenter.augment(frag_a)
            if np.random.random() < 0.5:
                frag_b = self.augmenter.augment(frag_b)
            # Point dropout with lower probability
            if np.random.random() < 0.2:
                frag_a = self.augmenter.dropout_points(frag_a, dropout_rate=0.1)
            if np.random.random() < 0.2:
                frag_b = self.augmenter.dropout_points(frag_b, dropout_rate=0.1)

        # Extract sequences
        seq_a = self._extract_sequence(frag_a)
        seq_b = self._extract_sequence(frag_b)

        # Extract endpoint features
        endpoint_features = self._extract_endpoint_features(frag_a, frag_b)

        # Normalize sequences
        if self.normalize:
            seq_a = self._normalize_sequence(seq_a)
            seq_b = self._normalize_sequence(seq_b)

        # Convert to tensors
        seq_a = torch.FloatTensor(seq_a)
        seq_b = torch.FloatTensor(seq_b)
        endpoint_features = torch.FloatTensor(endpoint_features)
        label = torch.FloatTensor([label])

        return seq_a, seq_b, endpoint_features, label


if __name__ == "__main__":
    # Test the dataset
    print("Testing TrajectoryPairDataset...")

    dataset = TrajectoryPairDataset(
        dataset_names=['i'],  # Start with one dataset for testing
        normalize=True
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test one sample
    seq_a, seq_b, endpoint_features, label = dataset[0]
    print(f"\nSample 0:")
    print(f"  Sequence A shape: {seq_a.shape}")
    print(f"  Sequence B shape: {seq_b.shape}")
    print(f"  Endpoint features: {endpoint_features}")
    print(f"  Label: {label.item()}")

    # Test DataLoader with custom collate
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(loader))
    padded_a, lengths_a, padded_b, lengths_b, endpoint_feats, labels = batch

    print(f"\nBatch test:")
    print(f"  Padded A shape: {padded_a.shape}")
    print(f"  Lengths A: {lengths_a}")
    print(f"  Padded B shape: {padded_b.shape}")
    print(f"  Lengths B: {lengths_b}")
    print(f"  Endpoint features shape: {endpoint_feats.shape}")
    print(f"  Labels: {labels.squeeze()}")

    # Test enhanced dataset if available
    if ENHANCED_PIPELINE_AVAILABLE:
        print("\n" + "="*60)
        print("Testing EnhancedTrajectoryPairDataset...")
        try:
            enhanced_dataset = EnhancedTrajectoryPairDataset(
                dataset_names=['i'],
                use_masking=True,
                use_hard_negatives=True,
                augment=False
            )
            print(f"\nEnhanced dataset size: {len(enhanced_dataset)}")
            print("EnhancedTrajectoryPairDataset OK")
        except Exception as e:
            print(f"Enhanced dataset test failed: {e}")
