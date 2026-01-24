"""
Trajectory Masking for Synthetic Positive Pair Generation

Takes complete GT trajectories and introduces masks (gaps) at varying locations
to create positive pairs that simulate real fragmentation patterns at camera handoffs.

This dramatically increases training data from ~2K to ~15K positive pairs.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import random


@dataclass
class MaskConfig:
    """Configuration for trajectory masking"""
    min_mask_length: float = 0.5      # Minimum mask duration in seconds
    max_mask_length: float = 3.0      # Maximum mask duration in seconds
    min_fragment_length: int = 15     # Minimum points in a fragment (~0.6s at 25fps)
    camera_boundaries: List[float] = field(default_factory=lambda: list(range(-200, 2000, 400)))
    boundary_tolerance: float = 50.0  # Tolerance around camera boundaries (feet)
    curriculum_levels: int = 5        # Number of difficulty levels for curriculum learning


class TrajectoryMasker:
    """
    Creates positive pairs from GT trajectories by introducing masks.

    Key features:
    - Realistic mask locations near camera boundaries (~400ft intervals)
    - Variable mask lengths for curriculum learning
    - Kinematic-aware augmentation
    - Preserves ground truth labels
    """

    def __init__(self, config: MaskConfig = None):
        """
        Args:
            config: Masking configuration
        """
        self.config = config or MaskConfig()

    def find_mask_candidates(self, trajectory: Dict) -> List[int]:
        """
        Find candidate indices for mask insertion near camera boundaries.

        Args:
            trajectory: GT trajectory dictionary

        Returns:
            List of indices where masks could be inserted
        """
        x_positions = np.array(trajectory['x_position'])
        candidates = []

        for boundary in self.config.camera_boundaries:
            # Find points near this camera boundary
            distances = np.abs(x_positions - boundary)
            near_boundary = np.where(distances < self.config.boundary_tolerance)[0]

            if len(near_boundary) > 0:
                # Choose the point closest to the boundary
                best_idx = near_boundary[np.argmin(distances[near_boundary])]
                # Ensure enough points on both sides
                if (best_idx >= self.config.min_fragment_length and
                    len(x_positions) - best_idx >= self.config.min_fragment_length):
                    candidates.append(best_idx)

        return sorted(set(candidates))

    def create_mask(self,
                    trajectory: Dict,
                    mask_start_idx: int,
                    mask_duration: float) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        """
        Create two fragments from a trajectory by masking a region.

        Args:
            trajectory: Complete GT trajectory
            mask_start_idx: Index where mask begins
            mask_duration: Duration of mask in seconds

        Returns:
            Tuple of (fragment_before, fragment_after, mask_metadata) or (None, None, None)
        """
        timestamps = np.array(trajectory['timestamp'])
        mask_start_time = timestamps[mask_start_idx]
        mask_end_time = mask_start_time + mask_duration

        # Find mask end index
        mask_end_idx = np.searchsorted(timestamps, mask_end_time)

        # Ensure valid fragments on both sides
        if mask_start_idx < self.config.min_fragment_length:
            return None, None, None
        if len(timestamps) - mask_end_idx < self.config.min_fragment_length:
            return None, None, None

        # Extract fields that are per-frame arrays
        array_fields = ['timestamp', 'x_position', 'y_position', 'velocity',
                       'detection_confidence', 'length', 'width', 'height']

        # Create fragment before mask
        frag_before = self._extract_fragment(trajectory, 0, mask_start_idx, array_fields)

        # Create fragment after mask
        frag_after = self._extract_fragment(trajectory, mask_end_idx, len(timestamps), array_fields)

        if frag_before is None or frag_after is None:
            return None, None, None

        # Metadata about the mask
        mask_metadata = {
            'gt_id': trajectory.get('_id', {}).get('$oid', 'unknown') if isinstance(trajectory.get('_id'), dict) else str(trajectory.get('_id', 'unknown')),
            'mask_start_time': float(mask_start_time),
            'mask_end_time': float(mask_end_time),
            'mask_duration': mask_duration,
            'mask_start_x': float(trajectory['x_position'][mask_start_idx]),
            'gap_points': mask_end_idx - mask_start_idx
        }

        return frag_before, frag_after, mask_metadata

    def _extract_fragment(self,
                         trajectory: Dict,
                         start_idx: int,
                         end_idx: int,
                         array_fields: List[str]) -> Optional[Dict]:
        """Extract a fragment from trajectory between indices."""
        if end_idx <= start_idx:
            return None

        fragment = {}

        for field in array_fields:
            if field in trajectory:
                data = trajectory[field]
                if isinstance(data, list):
                    fragment[field] = data[start_idx:end_idx]
                elif isinstance(data, np.ndarray):
                    fragment[field] = data[start_idx:end_idx].tolist()

        # Skip if no valid data
        if 'timestamp' not in fragment or len(fragment['timestamp']) == 0:
            return None

        # Add derived fields
        fragment['first_timestamp'] = fragment['timestamp'][0]
        fragment['last_timestamp'] = fragment['timestamp'][-1]

        if 'x_position' in fragment:
            fragment['starting_x'] = fragment['x_position'][0]
            fragment['ending_x'] = fragment['x_position'][-1]

        # Copy scalar fields
        scalar_fields = ['direction', 'coarse_vehicle_class', 'fine_vehicle_class']
        for field in scalar_fields:
            if field in trajectory:
                fragment[field] = trajectory[field]

        # Add GT ID for verification
        gt_id = trajectory.get('_id', {})
        if isinstance(gt_id, dict) and '$oid' in gt_id:
            fragment['_source_gt_id'] = gt_id['$oid']
        else:
            fragment['_source_gt_id'] = str(gt_id)

        return fragment

    def generate_curriculum_pairs(self,
                                  trajectory: Dict,
                                  difficulty_level: int) -> List[Tuple[Dict, Dict, Dict]]:
        """
        Generate pairs with mask length based on curriculum difficulty.

        Args:
            trajectory: GT trajectory
            difficulty_level: 0 (easy) to curriculum_levels-1 (hard)

        Returns:
            List of (frag_before, frag_after, metadata) tuples
        """
        # Map difficulty to mask duration
        level_frac = difficulty_level / max(1, self.config.curriculum_levels - 1)
        min_dur = self.config.min_mask_length
        max_dur = self.config.max_mask_length
        mask_duration = min_dur + level_frac * (max_dur - min_dur)

        # Add some noise to duration
        mask_duration *= np.random.uniform(0.9, 1.1)

        candidates = self.find_mask_candidates(trajectory)
        pairs = []

        for idx in candidates:
            result = self.create_mask(trajectory, idx, mask_duration)
            if result[0] is not None:
                pairs.append(result)

        return pairs

    def generate_all_pairs(self,
                          gt_trajectories: List[Dict],
                          pairs_per_trajectory: int = 3,
                          include_random_masks: bool = True) -> List[Tuple[Dict, Dict, int]]:
        """
        Generate positive pairs from all GT trajectories.

        Args:
            gt_trajectories: List of GT trajectory dictionaries
            pairs_per_trajectory: Target number of pairs per trajectory
            include_random_masks: Also include masks at random locations

        Returns:
            List of (fragment_a, fragment_b, label=1) tuples
        """
        all_pairs = []

        for traj in gt_trajectories:
            # Skip short trajectories
            if 'timestamp' not in traj or len(traj['timestamp']) < 2 * self.config.min_fragment_length:
                continue

            # Generate pairs at camera boundaries (most realistic)
            for level in range(self.config.curriculum_levels):
                curriculum_pairs = self.generate_curriculum_pairs(traj, level)
                for frag_a, frag_b, meta in curriculum_pairs:
                    all_pairs.append((frag_a, frag_b, 1))  # Label 1 = positive

            # Also generate random mask pairs if requested
            if include_random_masks:
                random_pairs = self._generate_random_mask_pairs(traj, pairs_per_trajectory)
                all_pairs.extend(random_pairs)

        return all_pairs

    def _generate_random_mask_pairs(self,
                                    trajectory: Dict,
                                    num_pairs: int) -> List[Tuple[Dict, Dict, int]]:
        """Generate pairs with masks at random locations."""
        pairs = []
        timestamps = trajectory['timestamp']
        n_points = len(timestamps)

        for _ in range(num_pairs):
            # Random mask start (ensure enough points on both sides)
            max_start = n_points - 2 * self.config.min_fragment_length
            if max_start <= self.config.min_fragment_length:
                continue

            mask_start_idx = random.randint(
                self.config.min_fragment_length,
                max_start
            )

            # Random mask duration
            mask_duration = random.uniform(
                self.config.min_mask_length,
                self.config.max_mask_length
            )

            result = self.create_mask(trajectory, mask_start_idx, mask_duration)
            if result[0] is not None:
                pairs.append((result[0], result[1], 1))

        return pairs


class KinematicAugmenter:
    """
    Apply kinematic-aware augmentations to trajectory fragments.

    Augmentations are physics-realistic:
    - Velocity variations within realistic bounds
    - Small lane changes (y-position variations)
    - Timestamp jitter (simulating detection timing variations)
    """

    def __init__(self,
                 velocity_noise_std: float = 2.0,  # ft/s (~5% at 40 ft/s)
                 y_noise_std: float = 0.5,         # feet
                 time_jitter_std: float = 0.02,    # seconds
                 x_noise_std: float = 1.0):        # feet
        """
        Args:
            velocity_noise_std: Standard deviation for velocity noise
            y_noise_std: Standard deviation for y-position noise
            time_jitter_std: Standard deviation for timestamp jitter
            x_noise_std: Standard deviation for x-position noise
        """
        self.velocity_noise_std = velocity_noise_std
        self.y_noise_std = y_noise_std
        self.time_jitter_std = time_jitter_std
        self.x_noise_std = x_noise_std

    def augment(self, fragment: Dict, augment_types: List[str] = None) -> Dict:
        """
        Apply augmentations to a fragment.

        Args:
            fragment: Fragment dictionary
            augment_types: List of augmentation types to apply
                          Options: 'velocity', 'y_position', 'time', 'x_position', 'dropout'

        Returns:
            Augmented fragment (copy)
        """
        if augment_types is None:
            augment_types = ['velocity', 'y_position', 'time', 'x_position']

        # Create a deep copy
        aug_frag = {}
        for k, v in fragment.items():
            if isinstance(v, list):
                aug_frag[k] = list(v)
            elif isinstance(v, np.ndarray):
                aug_frag[k] = v.copy().tolist()
            else:
                aug_frag[k] = v

        if 'timestamp' not in aug_frag:
            return aug_frag

        n_points = len(aug_frag['timestamp'])

        if 'velocity' in augment_types and 'velocity' in aug_frag and len(aug_frag['velocity']) > 0:
            noise = np.random.normal(0, self.velocity_noise_std, n_points)
            aug_frag['velocity'] = [float(v + n) for v, n in zip(aug_frag['velocity'], noise)]

        if 'y_position' in augment_types and 'y_position' in aug_frag and len(aug_frag['y_position']) > 0:
            # Correlated noise (not independent per point)
            base_noise = np.random.normal(0, self.y_noise_std)
            noise = base_noise + np.random.normal(0, self.y_noise_std * 0.1, n_points)
            aug_frag['y_position'] = [float(y + n) for y, n in zip(aug_frag['y_position'], noise)]

        if 'time' in augment_types and 'timestamp' in aug_frag and len(aug_frag['timestamp']) > 0:
            # Small timing jitter (keeps order)
            noise = np.random.normal(0, self.time_jitter_std, n_points)
            noise = np.cumsum(noise) * 0.1  # Make it smooth
            aug_frag['timestamp'] = [float(t + n) for t, n in zip(aug_frag['timestamp'], noise)]
            aug_frag['first_timestamp'] = aug_frag['timestamp'][0]
            aug_frag['last_timestamp'] = aug_frag['timestamp'][-1]

        if 'x_position' in augment_types and 'x_position' in aug_frag and len(aug_frag['x_position']) > 0:
            # Correlated noise
            noise = np.random.normal(0, self.x_noise_std, n_points)
            noise = np.cumsum(noise) * 0.1
            aug_frag['x_position'] = [float(x + n) for x, n in zip(aug_frag['x_position'], noise)]
            aug_frag['starting_x'] = aug_frag['x_position'][0]
            aug_frag['ending_x'] = aug_frag['x_position'][-1]

        return aug_frag

    def dropout_points(self, fragment: Dict, dropout_rate: float = 0.1) -> Dict:
        """
        Randomly drop points from fragment (simulates missed detections).

        Args:
            fragment: Fragment dictionary
            dropout_rate: Fraction of points to drop

        Returns:
            Fragment with some points dropped
        """
        if 'timestamp' not in fragment:
            return fragment

        n_points = len(fragment['timestamp'])
        keep_mask = np.random.random(n_points) > dropout_rate

        # Ensure we keep at least min_points
        min_points = 10
        if keep_mask.sum() < min_points:
            # Force keep first and last few points
            keep_mask[:5] = True
            keep_mask[-5:] = True

        aug_frag = {}
        array_fields = ['timestamp', 'x_position', 'y_position', 'velocity',
                       'detection_confidence', 'length', 'width', 'height']

        for field in array_fields:
            if field in fragment and isinstance(fragment[field], list) and len(fragment[field]) == n_points:
                aug_frag[field] = [v for v, keep in zip(fragment[field], keep_mask) if keep]

        # Copy other fields
        for k, v in fragment.items():
            if k not in aug_frag:
                aug_frag[k] = v

        # Update derived fields
        if 'timestamp' in aug_frag and len(aug_frag['timestamp']) > 0:
            aug_frag['first_timestamp'] = aug_frag['timestamp'][0]
            aug_frag['last_timestamp'] = aug_frag['timestamp'][-1]
        if 'x_position' in aug_frag and len(aug_frag['x_position']) > 0:
            aug_frag['starting_x'] = aug_frag['x_position'][0]
            aug_frag['ending_x'] = aug_frag['x_position'][-1]

        return aug_frag


if __name__ == "__main__":
    # Test the masking module
    print("Testing TrajectoryMasker...")

    # Create a dummy GT trajectory
    n_points = 200  # ~8 seconds at 25fps
    timestamps = np.linspace(0, 8, n_points)
    x_positions = np.linspace(0, 600, n_points)  # 600 ft in 8 seconds ~ 75 ft/s
    y_positions = np.ones(n_points) * 6  # Fixed lane
    velocities = np.ones(n_points) * 75

    dummy_trajectory = {
        '_id': {'$oid': 'test_vehicle_001'},
        'timestamp': timestamps.tolist(),
        'x_position': x_positions.tolist(),
        'y_position': y_positions.tolist(),
        'velocity': velocities.tolist(),
        'direction': 1
    }

    # Test masking
    masker = TrajectoryMasker()

    # Find candidates
    candidates = masker.find_mask_candidates(dummy_trajectory)
    print(f"Found {len(candidates)} mask candidates at indices: {candidates}")

    # Generate pairs
    pairs = masker.generate_all_pairs([dummy_trajectory], pairs_per_trajectory=3)
    print(f"Generated {len(pairs)} positive pairs")

    if len(pairs) > 0:
        frag_a, frag_b, label = pairs[0]
        print(f"\nSample pair:")
        print(f"  Fragment A: {len(frag_a['timestamp'])} points, "
              f"t=[{frag_a['first_timestamp']:.2f}, {frag_a['last_timestamp']:.2f}]")
        print(f"  Fragment B: {len(frag_b['timestamp'])} points, "
              f"t=[{frag_b['first_timestamp']:.2f}, {frag_b['last_timestamp']:.2f}]")
        print(f"  Gap: {frag_b['first_timestamp'] - frag_a['last_timestamp']:.2f}s")
        print(f"  Label: {label}")

    # Test augmenter
    print("\nTesting KinematicAugmenter...")
    augmenter = KinematicAugmenter()

    if len(pairs) > 0:
        aug_frag = augmenter.augment(pairs[0][0])
        print(f"Original velocity mean: {np.mean(pairs[0][0]['velocity']):.2f}")
        print(f"Augmented velocity mean: {np.mean(aug_frag['velocity']):.2f}")

        dropped_frag = augmenter.dropout_points(pairs[0][0], dropout_rate=0.2)
        print(f"Original points: {len(pairs[0][0]['timestamp'])}")
        print(f"After dropout: {len(dropped_frag['timestamp'])}")

    print("\nTrajectoryMasker OK")
