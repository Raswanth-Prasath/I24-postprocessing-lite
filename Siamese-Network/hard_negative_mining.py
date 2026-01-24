"""
Hard Negative Mining for Siamese Network Training

Generates challenging negative pairs from different vehicles that are:
- In the same lane (similar y-position)
- Close in time (small gap)
- Similar in appearance (length, velocity)

Uses finer spatial-temporal binning (0.5s, 50ft) vs original (2s, 150ft)
to capture truly hard negatives.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import random


@dataclass
class HardNegativeConfig:
    """Configuration for hard negative mining"""
    # Spatial-temporal binning for coarse filtering (finer than original 2s/150ft)
    time_bin_size: float = 0.5       # seconds
    x_bin_size: float = 50.0         # feet

    # Constraints for valid negative pairs
    max_time_gap: float = 5.0        # Maximum time gap between fragments
    max_y_diff: float = 8.0          # Maximum y-position difference (same lane ~12ft)
    max_velocity_diff: float = 30.0  # Maximum velocity difference for hard negatives

    # Sampling parameters
    negatives_per_positive: int = 3  # Number of negatives per positive
    hard_ratio: float = 0.7          # Fraction of negatives that are "hard"


class HardNegativeMiner:
    """
    Mines hard negative pairs from RAW fragments using GT labels.

    Hard negatives are pairs that:
    1. Come from DIFFERENT vehicles (verified by gt_ids)
    2. Are spatially close (same lane, small x gap)
    3. Are temporally close (small time gap)
    4. Have similar kinematics (velocity, length)
    """

    def __init__(self, config: HardNegativeConfig = None):
        """
        Args:
            config: Configuration for mining
        """
        self.config = config or HardNegativeConfig()

        # Index structures for efficient lookup
        self.spatial_temporal_index = defaultdict(list)
        self.vehicle_index = defaultdict(list)  # gt_id -> list of fragment indices
        self.fragments = []

    def build_index(self, raw_fragments: List[Dict]):
        """
        Build spatial-temporal and vehicle indices for efficient mining.

        Args:
            raw_fragments: List of RAW fragment dictionaries
        """
        self.fragments = raw_fragments
        self.spatial_temporal_index.clear()
        self.vehicle_index.clear()

        for idx, frag in enumerate(raw_fragments):
            # Get GT vehicle ID
            gt_id = self._get_gt_id(frag)
            if gt_id is not None:
                self.vehicle_index[gt_id].append(idx)

            # Add to spatial-temporal index
            first_ts = frag.get('first_timestamp', frag['timestamp'][0] if 'timestamp' in frag else 0)
            starting_x = frag.get('starting_x', frag['x_position'][0] if 'x_position' in frag else 0)

            t_bin = int(first_ts / self.config.time_bin_size)
            x_bin = int(starting_x / self.config.x_bin_size)
            direction = frag.get('direction', 1)

            key = (t_bin, x_bin, direction)
            self.spatial_temporal_index[key].append(idx)

    def _get_gt_id(self, fragment: Dict) -> Optional[str]:
        """Extract ground truth vehicle ID from fragment."""
        if 'gt_ids' in fragment and len(fragment['gt_ids']) > 0:
            first_gt_id = fragment['gt_ids'][0]
            if isinstance(first_gt_id, list) and len(first_gt_id) > 0:
                if isinstance(first_gt_id[0], dict) and '$oid' in first_gt_id[0]:
                    return first_gt_id[0]['$oid']
        # Also check for _source_gt_id (from masking)
        if '_source_gt_id' in fragment:
            return fragment['_source_gt_id']
        return None

    def _get_fragment_timestamps(self, frag: Dict) -> Tuple[float, float]:
        """Get first and last timestamps from fragment."""
        first_ts = frag.get('first_timestamp')
        last_ts = frag.get('last_timestamp')

        if first_ts is None and 'timestamp' in frag and len(frag['timestamp']) > 0:
            first_ts = frag['timestamp'][0]
        if last_ts is None and 'timestamp' in frag and len(frag['timestamp']) > 0:
            last_ts = frag['timestamp'][-1]

        return first_ts or 0, last_ts or 0

    def _is_valid_negative_pair(self,
                                frag_a: Dict,
                                frag_b: Dict) -> bool:
        """
        Check if two fragments form a valid negative pair.

        Valid negative = different vehicles, but could plausibly be same vehicle
        based on spatial-temporal proximity.
        """
        # Must be different vehicles
        gt_a = self._get_gt_id(frag_a)
        gt_b = self._get_gt_id(frag_b)

        if gt_a is None or gt_b is None:
            return False
        if gt_a == gt_b:
            return False  # Same vehicle = positive, not negative

        # Get timestamps
        _, last_a = self._get_fragment_timestamps(frag_a)
        first_b, _ = self._get_fragment_timestamps(frag_b)

        # Must be sequential (frag_b comes after frag_a)
        if last_a >= first_b:
            return False

        # Must have same direction
        if frag_a.get('direction') != frag_b.get('direction'):
            return False

        # Time gap constraint
        time_gap = first_b - last_a
        if time_gap > self.config.max_time_gap:
            return False

        return True

    def _compute_hardness_score(self,
                                frag_a: Dict,
                                frag_b: Dict) -> float:
        """
        Compute how "hard" a negative pair is.

        Higher score = harder negative (more similar to positive pairs)

        Returns:
            Hardness score in [0, 1]
        """
        scores = []

        # Y-position similarity (same lane = harder)
        y_a = np.mean(frag_a['y_position']) if 'y_position' in frag_a else 0
        y_b = np.mean(frag_b['y_position']) if 'y_position' in frag_b else 0
        y_diff = abs(y_a - y_b)
        y_score = max(0, 1 - y_diff / self.config.max_y_diff)
        scores.append(y_score)

        # Velocity similarity
        if 'velocity' in frag_a and 'velocity' in frag_b and len(frag_a['velocity']) > 0 and len(frag_b['velocity']) > 0:
            v_a = np.mean(frag_a['velocity'])
            v_b = np.mean(frag_b['velocity'])
            v_diff = abs(v_a - v_b)
            v_score = max(0, 1 - v_diff / self.config.max_velocity_diff)
            scores.append(v_score)

        # Time gap (smaller = harder)
        _, last_a = self._get_fragment_timestamps(frag_a)
        first_b, _ = self._get_fragment_timestamps(frag_b)
        time_gap = first_b - last_a
        time_score = max(0, 1 - time_gap / self.config.max_time_gap)
        scores.append(time_score)

        # X-position continuity (smaller gap = harder)
        direction = frag_a.get('direction', 1)
        ending_x_a = frag_a.get('ending_x', frag_a['x_position'][-1] if 'x_position' in frag_a else 0)
        starting_x_b = frag_b.get('starting_x', frag_b['x_position'][0] if 'x_position' in frag_b else 0)

        if direction == 1:  # Eastbound
            x_gap = starting_x_b - ending_x_a
        else:  # Westbound
            x_gap = ending_x_a - starting_x_b

        # Expected x-gap based on velocity and time
        if 'velocity' in frag_a and len(frag_a['velocity']) > 0:
            expected_gap = abs(np.mean(frag_a['velocity'])) * time_gap
        else:
            expected_gap = 50 * time_gap  # Default 50 ft/s

        gap_diff = abs(x_gap - expected_gap)
        gap_score = max(0, 1 - gap_diff / 100)  # 100ft tolerance
        scores.append(gap_score)

        # Vehicle dimensions similarity
        if 'length' in frag_a and 'length' in frag_b and len(frag_a['length']) > 0 and len(frag_b['length']) > 0:
            len_a = np.mean(frag_a['length'])
            len_b = np.mean(frag_b['length'])
            len_diff = abs(len_a - len_b)
            len_score = max(0, 1 - len_diff / 10)  # 10ft max diff
            scores.append(len_score)

        return np.mean(scores) if scores else 0.0

    def mine_hard_negatives(self,
                           num_negatives: int,
                           min_hardness: float = 0.3) -> List[Tuple[int, int, float]]:
        """
        Mine hard negative pairs from indexed fragments.

        Args:
            num_negatives: Number of negative pairs to generate
            min_hardness: Minimum hardness score for a pair to be considered

        Returns:
            List of (idx_a, idx_b, hardness_score) tuples
        """
        hard_negatives = []
        search_radius = 3  # Search in neighboring bins
        seen_pairs = set()

        # Iterate through spatial-temporal bins
        for (t_bin, x_bin, direction), indices in self.spatial_temporal_index.items():
            if len(hard_negatives) >= num_negatives * 2:  # Collect extras for sorting
                break

            # Look for candidates in neighboring bins (later in time)
            candidate_indices = set()
            for dt in range(0, search_radius + 1):  # Only forward in time
                for dx in range(-search_radius, search_radius + 1):
                    neighbor_key = (t_bin + dt, x_bin + dx, direction)
                    if neighbor_key in self.spatial_temporal_index:
                        candidate_indices.update(self.spatial_temporal_index[neighbor_key])

            # Check all pairs
            for idx_a in indices:
                for idx_b in candidate_indices:
                    if idx_a == idx_b:
                        continue

                    # Create canonical pair key to avoid duplicates
                    pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    frag_a = self.fragments[idx_a]
                    frag_b = self.fragments[idx_b]

                    # Ensure correct ordering (a before b)
                    _, last_a = self._get_fragment_timestamps(frag_a)
                    first_b, _ = self._get_fragment_timestamps(frag_b)

                    if last_a > first_b:
                        idx_a, idx_b = idx_b, idx_a
                        frag_a, frag_b = frag_b, frag_a

                    if not self._is_valid_negative_pair(frag_a, frag_b):
                        continue

                    hardness = self._compute_hardness_score(frag_a, frag_b)

                    if hardness >= min_hardness:
                        hard_negatives.append((idx_a, idx_b, hardness))

        # Sort by hardness (descending) and return top negatives
        hard_negatives.sort(key=lambda x: x[2], reverse=True)
        return hard_negatives[:num_negatives]

    def mine_negatives_balanced(self,
                               num_negatives: int,
                               hard_ratio: float = 0.7) -> List[Tuple[int, int, float]]:
        """
        Mine a balanced mix of hard and easier negatives.

        Args:
            num_negatives: Total number of negatives to generate
            hard_ratio: Fraction of hard negatives (hardness > 0.5)

        Returns:
            List of (idx_a, idx_b, hardness_score) tuples
        """
        # Mine all candidates with low threshold
        all_negatives = self.mine_hard_negatives(num_negatives * 3, min_hardness=0.1)

        if not all_negatives:
            return []

        # Split into hard and easy
        hard = [n for n in all_negatives if n[2] >= 0.5]
        easy = [n for n in all_negatives if n[2] < 0.5]

        # Sample according to ratio
        num_hard = int(num_negatives * hard_ratio)
        num_easy = num_negatives - num_hard

        selected_hard = hard[:num_hard] if len(hard) >= num_hard else hard
        selected_easy = easy[:num_easy] if len(easy) >= num_easy else easy

        result = selected_hard + selected_easy
        random.shuffle(result)
        return result[:num_negatives]

    def mine_semi_hard_negatives(self,
                                 positive_pairs: List[Tuple[Dict, Dict]],
                                 num_negatives_per_positive: int = 3) -> List[Tuple[Dict, Dict, Dict]]:
        """
        Mine semi-hard negatives for triplet loss training.

        For each positive pair (anchor, positive), find negatives that are:
        - Harder than easy negatives (different lane, far away)
        - But not so hard they're ambiguous

        Args:
            positive_pairs: List of (anchor_fragment, positive_fragment) tuples
            num_negatives_per_positive: Negatives per positive pair

        Returns:
            List of (anchor, positive, negative) triplets
        """
        triplets = []

        for anchor, positive in positive_pairs:
            anchor_gt = self._get_gt_id(anchor)

            # Find candidates near the positive (where we'd expect the anchor to continue)
            first_ts, _ = self._get_fragment_timestamps(positive)
            starting_x = positive.get('starting_x', positive['x_position'][0] if 'x_position' in positive else 0)

            t_bin = int(first_ts / self.config.time_bin_size)
            x_bin = int(starting_x / self.config.x_bin_size)
            direction = positive.get('direction', 1)

            # Search for negatives in nearby bins
            candidates = []
            for dt in range(-2, 3):
                for dx in range(-2, 3):
                    key = (t_bin + dt, x_bin + dx, direction)
                    if key in self.spatial_temporal_index:
                        candidates.extend(self.spatial_temporal_index[key])

            # Filter and score candidates
            negative_candidates = []
            for neg_idx in candidates:
                neg = self.fragments[neg_idx]
                neg_gt = self._get_gt_id(neg)

                # Must be different vehicle
                if neg_gt == anchor_gt or neg_gt is None:
                    continue

                # Must be sequential with anchor
                if not self._is_valid_negative_pair(anchor, neg):
                    continue

                hardness = self._compute_hardness_score(anchor, neg)
                negative_candidates.append((neg, hardness))

            if not negative_candidates:
                continue

            # Sort by hardness and select semi-hard negatives
            # (medium hardness, not too easy, not too hard)
            negative_candidates.sort(key=lambda x: x[1])
            n_candidates = len(negative_candidates)

            # Select from middle range (semi-hard)
            start = n_candidates // 4
            end = max(start + 1, 3 * n_candidates // 4)
            semi_hard = negative_candidates[start:end]

            if not semi_hard:
                semi_hard = negative_candidates  # Fall back to all candidates

            # Sample negatives
            num_to_sample = min(num_negatives_per_positive, len(semi_hard))
            selected = random.sample(semi_hard, num_to_sample)

            for neg, hardness in selected:
                triplets.append((anchor, positive, neg))

        return triplets

    def get_fragment_pairs(self,
                          negative_indices: List[Tuple[int, int, float]]) -> List[Tuple[Dict, Dict, int]]:
        """
        Convert index-based negatives to fragment pairs.

        Args:
            negative_indices: List of (idx_a, idx_b, hardness) tuples

        Returns:
            List of (fragment_a, fragment_b, label=0) tuples
        """
        pairs = []
        for idx_a, idx_b, _ in negative_indices:
            pairs.append((self.fragments[idx_a], self.fragments[idx_b], 0))
        return pairs


class TripletDataset:
    """
    Dataset that provides (anchor, positive, negative) triplets for triplet loss.
    """

    def __init__(self,
                 raw_fragments: List[Dict],
                 positive_pairs: List[Tuple[Dict, Dict]],
                 config: HardNegativeConfig = None):
        """
        Args:
            raw_fragments: List of RAW fragment dictionaries
            positive_pairs: List of (anchor, positive) fragment pairs
            config: Configuration for mining
        """
        self.fragments = raw_fragments
        self.positive_pairs = positive_pairs

        # Build miner and generate triplets
        self.miner = HardNegativeMiner(config)
        self.miner.build_index(raw_fragments)

        self.triplets = self.miner.mine_semi_hard_negatives(
            positive_pairs,
            num_negatives_per_positive=3
        )

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


if __name__ == "__main__":
    # Test the hard negative mining module
    print("Testing HardNegativeMiner...")

    # Create dummy fragments from two different vehicles
    np.random.seed(42)

    # Vehicle 1: travels from x=0 to x=300 over 6 seconds
    # Split into 2 fragments
    n_points = 75  # ~3 seconds each at 25fps

    vehicle1_frag1 = {
        '_id': 'v1_f1',
        'gt_ids': [[{'$oid': 'vehicle_001'}]],
        'timestamp': np.linspace(0, 3, n_points).tolist(),
        'x_position': np.linspace(0, 150, n_points).tolist(),
        'y_position': (np.ones(n_points) * 6 + np.random.normal(0, 0.2, n_points)).tolist(),
        'velocity': (np.ones(n_points) * 50 + np.random.normal(0, 2, n_points)).tolist(),
        'length': (np.ones(n_points) * 15).tolist(),
        'first_timestamp': 0,
        'last_timestamp': 3,
        'starting_x': 0,
        'ending_x': 150,
        'direction': 1
    }

    vehicle1_frag2 = {
        '_id': 'v1_f2',
        'gt_ids': [[{'$oid': 'vehicle_001'}]],
        'timestamp': np.linspace(3.5, 6.5, n_points).tolist(),
        'x_position': np.linspace(175, 325, n_points).tolist(),
        'y_position': (np.ones(n_points) * 6 + np.random.normal(0, 0.2, n_points)).tolist(),
        'velocity': (np.ones(n_points) * 50 + np.random.normal(0, 2, n_points)).tolist(),
        'length': (np.ones(n_points) * 15).tolist(),
        'first_timestamp': 3.5,
        'last_timestamp': 6.5,
        'starting_x': 175,
        'ending_x': 325,
        'direction': 1
    }

    # Vehicle 2: travels in same lane, ~1 second behind
    vehicle2_frag1 = {
        '_id': 'v2_f1',
        'gt_ids': [[{'$oid': 'vehicle_002'}]],
        'timestamp': np.linspace(1, 4, n_points).tolist(),
        'x_position': np.linspace(0, 150, n_points).tolist(),
        'y_position': (np.ones(n_points) * 6.5 + np.random.normal(0, 0.2, n_points)).tolist(),
        'velocity': (np.ones(n_points) * 50 + np.random.normal(0, 2, n_points)).tolist(),
        'length': (np.ones(n_points) * 16).tolist(),
        'first_timestamp': 1,
        'last_timestamp': 4,
        'starting_x': 0,
        'ending_x': 150,
        'direction': 1
    }

    vehicle2_frag2 = {
        '_id': 'v2_f2',
        'gt_ids': [[{'$oid': 'vehicle_002'}]],
        'timestamp': np.linspace(4.5, 7.5, n_points).tolist(),
        'x_position': np.linspace(175, 325, n_points).tolist(),
        'y_position': (np.ones(n_points) * 6.5 + np.random.normal(0, 0.2, n_points)).tolist(),
        'velocity': (np.ones(n_points) * 50 + np.random.normal(0, 2, n_points)).tolist(),
        'length': (np.ones(n_points) * 16).tolist(),
        'first_timestamp': 4.5,
        'last_timestamp': 7.5,
        'starting_x': 175,
        'ending_x': 325,
        'direction': 1
    }

    # Vehicle 3: different lane (negative control)
    vehicle3_frag1 = {
        '_id': 'v3_f1',
        'gt_ids': [[{'$oid': 'vehicle_003'}]],
        'timestamp': np.linspace(0.5, 3.5, n_points).tolist(),
        'x_position': np.linspace(0, 150, n_points).tolist(),
        'y_position': (np.ones(n_points) * 18 + np.random.normal(0, 0.2, n_points)).tolist(),  # Different lane
        'velocity': (np.ones(n_points) * 55 + np.random.normal(0, 2, n_points)).tolist(),
        'length': (np.ones(n_points) * 20).tolist(),  # Truck
        'first_timestamp': 0.5,
        'last_timestamp': 3.5,
        'starting_x': 0,
        'ending_x': 150,
        'direction': 1
    }

    fragments = [vehicle1_frag1, vehicle1_frag2, vehicle2_frag1, vehicle2_frag2, vehicle3_frag1]

    # Build index
    miner = HardNegativeMiner()
    miner.build_index(fragments)

    print(f"\nBuilt index with {len(fragments)} fragments")
    print(f"Vehicle index: {dict((k, len(v)) for k, v in miner.vehicle_index.items())}")
    print(f"Spatial-temporal bins: {len(miner.spatial_temporal_index)}")

    # Mine hard negatives
    hard_negs = miner.mine_hard_negatives(num_negatives=10, min_hardness=0.3)
    print(f"\nFound {len(hard_negs)} hard negative pairs:")

    for idx_a, idx_b, hardness in hard_negs:
        frag_a = fragments[idx_a]
        frag_b = fragments[idx_b]
        gt_a = miner._get_gt_id(frag_a)
        gt_b = miner._get_gt_id(frag_b)
        print(f"  {gt_a[-3:]} -> {gt_b[-3:]}: hardness={hardness:.3f}")

    # Test balanced mining
    balanced = miner.mine_negatives_balanced(num_negatives=5)
    print(f"\nBalanced mining: {len(balanced)} negatives")

    # Test semi-hard triplets
    positive_pairs = [(vehicle1_frag1, vehicle1_frag2)]  # Same vehicle = positive
    triplets = miner.mine_semi_hard_negatives(positive_pairs)
    print(f"\nSemi-hard triplets: {len(triplets)}")

    for anchor, positive, negative in triplets[:3]:
        print(f"  Anchor: {miner._get_gt_id(anchor)[-3:]}, "
              f"Positive: {miner._get_gt_id(positive)[-3:]}, "
              f"Negative: {miner._get_gt_id(negative)[-3:]}")

    print("\nHardNegativeMiner OK")
