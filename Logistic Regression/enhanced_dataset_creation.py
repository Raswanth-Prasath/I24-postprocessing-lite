"""
Enhanced Dataset Creation for Fragment Association Learning

This script generates a comprehensive training dataset by:
1. Loading all three I24-3D datasets (i, ii, iii)
2. Generating positive pairs from same-vehicle fragments
3. Mining hard negative pairs (spatially/temporally close but different vehicles)
4. Extracting rich features for each fragment pair
5. Balancing the dataset and saving for training
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

# Dataset paths
DATASET_DIR = Path(r"D:\ASU Academics\Thesis & Research\01_Papers\Datasets\I24-3D")
DATASETS = ['i', 'ii', 'iii']

class DatasetGenerator:
    def __init__(self, max_time_gap=5.0, max_spatial_gap=200, y_threshold=5.0):
        """
        Args:
            max_time_gap: Maximum time gap between fragments to consider (seconds)
            max_spatial_gap: Maximum spatial gap between fragments (feet)
            y_threshold: Maximum lateral distance to consider same lane (feet)
        """
        self.max_time_gap = max_time_gap
        self.max_spatial_gap = max_spatial_gap
        self.y_threshold = y_threshold

    def load_dataset(self, dataset_name: str) -> Tuple[List[Dict], List[Dict]]:
        """Load GT and RAW data for a specific dataset"""
        gt_path = DATASET_DIR / f"GT_{dataset_name}.json"
        raw_path = DATASET_DIR / f"RAW_{dataset_name}.json"

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        with open(raw_path, 'r') as f:
            raw_data = json.load(f)

        print(f"Loaded {dataset_name}: {len(gt_data)} GT trajectories, {len(raw_data)} RAW fragments")
        return gt_data, raw_data

    def get_gt_id(self, fragment: Dict) -> str:
        """Extract ground truth vehicle ID from fragment"""
        if 'gt_ids' in fragment and len(fragment['gt_ids']) > 0:
            return fragment['gt_ids'][0][0]['$oid']
        return None

    def is_sequential(self, frag_a: Dict, frag_b: Dict) -> bool:
        """Check if frag_b comes after frag_a without overlap"""
        return frag_a['last_timestamp'] < frag_b['first_timestamp']

    def is_candidate_pair(self, frag_a: Dict, frag_b: Dict) -> bool:
        """Check if two fragments could potentially be stitched"""
        # Must be sequential
        if not self.is_sequential(frag_a, frag_b):
            return False

        # Must have same direction
        if frag_a['direction'] != frag_b['direction']:
            return False

        # Time gap constraint
        time_gap = frag_b['first_timestamp'] - frag_a['last_timestamp']
        if time_gap > self.max_time_gap:
            return False

        # Spatial gap constraint (for eastbound, positive gap expected)
        if frag_a['direction'] == 1:  # Eastbound
            spatial_gap = frag_b['starting_x'] - frag_a['ending_x']
        else:  # Westbound
            spatial_gap = frag_a['starting_x'] - frag_b['ending_x']

        if spatial_gap < -50 or spatial_gap > self.max_spatial_gap:
            return False

        # Y-position constraint (approximate same lane)
        y_diff = abs(np.mean(frag_b['y_position']) - np.mean(frag_a['y_position']))
        if y_diff > self.y_threshold:
            return False

        return True

    def extract_basic_features(self, frag_a: Dict, frag_b: Dict) -> Dict[str, float]:
        """Extract basic geometric and kinematic features"""
        features = {}

        # Temporal features
        features['time_gap'] = frag_b['first_timestamp'] - frag_a['last_timestamp']
        features['duration_a'] = frag_a['last_timestamp'] - frag_a['first_timestamp']
        features['duration_b'] = frag_b['last_timestamp'] - frag_b['first_timestamp']
        features['duration_ratio'] = features['duration_a'] / (features['duration_b'] + 1e-6)

        # Spatial features
        if frag_a['direction'] == 1:  # Eastbound
            features['spatial_gap'] = frag_b['starting_x'] - frag_a['ending_x']
        else:  # Westbound
            features['spatial_gap'] = frag_a['starting_x'] - frag_b['ending_x']

        # Lateral features
        features['y_mean_a'] = np.mean(frag_a['y_position'])
        features['y_mean_b'] = np.mean(frag_b['y_position'])
        features['y_diff'] = abs(features['y_mean_b'] - features['y_mean_a'])
        features['y_std_a'] = np.std(frag_a['y_position'])
        features['y_std_b'] = np.std(frag_b['y_position'])

        # Kinematic features
        if 'velocity' in frag_a and 'velocity' in frag_b:
            vel_a_end = frag_a['velocity'][-1] if len(frag_a['velocity']) > 0 else 0
            vel_b_start = frag_b['velocity'][0] if len(frag_b['velocity']) > 0 else 0
            features['vel_a_mean'] = np.mean(frag_a['velocity'])
            features['vel_b_mean'] = np.mean(frag_b['velocity'])
            features['vel_diff'] = abs(vel_a_end - vel_b_start)
            features['vel_ratio'] = vel_a_end / (vel_b_start + 1e-6)
        else:
            features['vel_a_mean'] = 0
            features['vel_b_mean'] = 0
            features['vel_diff'] = 0
            features['vel_ratio'] = 1.0

        # Vehicle dimension features
        features['length_a'] = np.mean(frag_a['length'])
        features['length_b'] = np.mean(frag_b['length'])
        features['length_diff'] = abs(features['length_a'] - features['length_b'])
        features['width_a'] = np.mean(frag_a['width'])
        features['width_b'] = np.mean(frag_b['width'])
        features['width_diff'] = abs(features['width_a'] - features['width_b'])

        if 'height' in frag_a and 'height' in frag_b:
            features['height_a'] = np.mean(frag_a['height'])
            features['height_b'] = np.mean(frag_b['height'])
            features['height_diff'] = abs(features['height_a'] - features['height_b'])
        else:
            features['height_a'] = 0
            features['height_b'] = 0
            features['height_diff'] = 0

        # Detection confidence features
        if 'detection_confidence' in frag_a and 'detection_confidence' in frag_b:
            features['conf_a_mean'] = np.mean(frag_a['detection_confidence'])
            features['conf_b_mean'] = np.mean(frag_b['detection_confidence'])
            features['conf_a_min'] = np.min(frag_a['detection_confidence'])
            features['conf_b_min'] = np.min(frag_b['detection_confidence'])
        else:
            features['conf_a_mean'] = 1.0
            features['conf_b_mean'] = 1.0
            features['conf_a_min'] = 1.0
            features['conf_b_min'] = 1.0

        # Direction feature (should always be 1 at this point due to filtering)
        features['direction_match'] = 1 if frag_a['direction'] == frag_b['direction'] else 0

        return features

    def generate_positive_pairs(self, raw_fragments: List[Dict]) -> List[Tuple[int, int, Dict]]:
        """Generate positive pairs (same vehicle)"""
        positive_pairs = []

        # Group fragments by GT vehicle ID
        vehicle_fragments = defaultdict(list)
        for idx, frag in enumerate(raw_fragments):
            gt_id = self.get_gt_id(frag)
            if gt_id is not None:
                vehicle_fragments[gt_id].append(idx)

        # Generate pairs within each vehicle
        for gt_id, fragment_indices in vehicle_fragments.items():
            for i, idx_a in enumerate(fragment_indices):
                for idx_b in fragment_indices[i+1:]:
                    frag_a = raw_fragments[idx_a]
                    frag_b = raw_fragments[idx_b]

                    if self.is_candidate_pair(frag_a, frag_b):
                        features = self.extract_basic_features(frag_a, frag_b)
                        positive_pairs.append((idx_a, idx_b, features))

        return positive_pairs

    def generate_hard_negatives(self, raw_fragments: List[Dict],
                                 num_negatives: int = None,
                                 relaxed_mode: bool = True) -> List[Tuple[int, int, Dict]]:
        """
        Generate hard negative pairs (different vehicles but spatially/temporally close)

        Args:
            num_negatives: Number of negative pairs to generate (if None, same as positives)
            relaxed_mode: If True, use relaxed constraints to generate more negatives
        """
        negative_pairs = []

        # Build spatial-temporal index for efficient lookup
        # Use larger bins for relaxed mode to capture more candidates
        time_window = 2.0 if relaxed_mode else 1.0  # seconds
        space_window = 150 if relaxed_mode else 100  # feet

        st_index = defaultdict(list)
        for idx, frag in enumerate(raw_fragments):
            t_bin = int(frag['first_timestamp'] / time_window)
            x_bin = int(frag['starting_x'] / space_window)
            direction = frag['direction']
            st_index[(t_bin, x_bin, direction)].append(idx)

        # Sample negative pairs from nearby fragments
        attempted = 0
        max_attempts = num_negatives * 20 if num_negatives else 200000

        # Expand search radius for relaxed mode
        search_radius = 2 if relaxed_mode else 1

        for (t_bin, x_bin, direction), fragment_indices in st_index.items():
            if num_negatives and len(negative_pairs) >= num_negatives:
                break

            # Look in neighboring bins with expanded radius
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

                            # Check if they're from different vehicles
                            gt_a = self.get_gt_id(frag_a)
                            gt_b = self.get_gt_id(frag_b)

                            if gt_a is None or gt_b is None or gt_a == gt_b:
                                continue

                            # In relaxed mode, use looser criteria
                            if relaxed_mode:
                                # Just check basic sequential and direction match
                                if not self.is_sequential(frag_a, frag_b):
                                    continue
                                if frag_a['direction'] != frag_b['direction']:
                                    continue

                                # Relaxed time/space constraints
                                time_gap = frag_b['first_timestamp'] - frag_a['last_timestamp']
                                if time_gap > self.max_time_gap * 2:  # 2x relaxed
                                    continue

                                features = self.extract_basic_features(frag_a, frag_b)
                                negative_pairs.append((idx_a, idx_b, features))
                            else:
                                # Strict mode: must be candidate pairs
                                if self.is_candidate_pair(frag_a, frag_b):
                                    features = self.extract_basic_features(frag_a, frag_b)
                                    negative_pairs.append((idx_a, idx_b, features))

                            if num_negatives and len(negative_pairs) >= num_negatives:
                                return negative_pairs

        return negative_pairs

    def create_dataset_from_scene(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create training dataset from a single scene"""
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        gt_data, raw_data = self.load_dataset(dataset_name)

        # Generate positive pairs
        print("Generating positive pairs...")
        positive_pairs = self.generate_positive_pairs(raw_data)
        print(f"Generated {len(positive_pairs)} positive pairs")

        # Generate hard negative pairs
        print("Mining hard negative pairs...")
        negative_pairs = self.generate_hard_negatives(raw_data, num_negatives=len(positive_pairs))
        print(f"Generated {len(negative_pairs)} negative pairs")

        # Combine and create arrays
        all_pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

        # Extract feature names from first pair
        if len(all_pairs) > 0:
            feature_names = list(all_pairs[0][2].keys())

            # Create feature matrix
            X = np.array([[pair[2][fname] for fname in feature_names] for pair in all_pairs])
            y = np.array(labels)

            print(f"\nDataset shape: {X.shape}")
            print(f"Positive pairs: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.2f}%)")
            print(f"Negative pairs: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.2f}%)")

            return X, y, feature_names
        else:
            return np.array([]), np.array([]), []

    def create_combined_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create combined training dataset from all three scenes"""
        print("\n" + "="*60)
        print("CREATING COMBINED DATASET FROM ALL SCENES")
        print("="*60)

        all_X = []
        all_y = []
        feature_names = None

        for dataset_name in DATASETS:
            X, y, fnames = self.create_dataset_from_scene(dataset_name)

            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

                if feature_names is None:
                    feature_names = fnames

        # Combine all datasets
        if len(all_X) > 0:
            X_combined = np.vstack(all_X)
            y_combined = np.hstack(all_y)

            print("\n" + "="*60)
            print("COMBINED DATASET STATISTICS")
            print("="*60)
            print(f"Total pairs: {len(y_combined)}")
            print(f"Positive pairs: {np.sum(y_combined == 1)} ({np.sum(y_combined == 1) / len(y_combined) * 100:.2f}%)")
            print(f"Negative pairs: {np.sum(y_combined == 0)} ({np.sum(y_combined == 0) / len(y_combined) * 100:.2f}%)")
            print(f"Feature dimensions: {X_combined.shape[1]}")
            print(f"\nFeature names ({len(feature_names)}):")
            for i, fname in enumerate(feature_names):
                print(f"  {i+1:2d}. {fname}")

            return X_combined, y_combined, feature_names
        else:
            return np.array([]), np.array([]), []

    def save_dataset(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                     output_path: str = "training_dataset.npz"):
        """Save dataset to disk"""
        # Use absolute path
        output_file = Path(r"D:\ASU Academics\Thesis & Research\02_Code\Logistic-Regression") / output_path
        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            feature_names=feature_names
        )
        print(f"\nDataset saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main execution function"""
    # Create dataset generator
    generator = DatasetGenerator(
        max_time_gap=5.0,
        max_spatial_gap=200,
        y_threshold=5.0
    )

    # Generate combined dataset
    X, y, feature_names = generator.create_combined_dataset()

    if len(X) > 0:
        # Save dataset
        generator.save_dataset(X, y, feature_names, "training_dataset_combined.npz")

        # Print basic statistics
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        df = pd.DataFrame(X, columns=feature_names)
        print(df.describe())

        print("\n" + "="*60)
        print("Dataset creation complete!")
        print("="*60)
    else:
        print("\nError: No data generated!")


if __name__ == "__main__":
    main()
