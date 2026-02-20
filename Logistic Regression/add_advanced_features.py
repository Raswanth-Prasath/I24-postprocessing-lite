"""
Add advanced features to existing dataset

Loads the basic dataset and adds Bhattacharyya distance and other sophisticated features
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from advanced_features import extract_advanced_features

# Paths (repo-relative by default)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT
WORKING_DIR = Path(__file__).resolve().parent
DATASETS = ['i', 'ii', 'iii']


def load_basic_dataset():
    """Load the basic dataset generated earlier"""
    data = np.load(WORKING_DIR / "training_dataset_combined.npz", allow_pickle=True)
    return data['X'], data['y'], list(data['feature_names'])


def load_raw_data():
    """Load all RAW fragment data"""
    all_fragments = []
    fragment_dataset_map = []  # Track which dataset each fragment belongs to

    for dataset_name in DATASETS:
        raw_path = DATASET_DIR / f"RAW_{dataset_name}.json"
        with open(raw_path, 'r') as f:
            frags = json.load(f)
            all_fragments.extend(frags)
            fragment_dataset_map.extend([dataset_name] * len(frags))

    return all_fragments, fragment_dataset_map


def get_gt_id(fragment):
    """Extract ground truth vehicle ID from fragment"""
    if 'gt_ids' in fragment and len(fragment['gt_ids']) > 0:
        return fragment['gt_ids'][0][0]['$oid']
    return None


def reconstruct_pairs_from_dataset(X_basic, y, all_fragments):
    """
    Reconstruct which fragment pairs correspond to each row in the dataset

    This is tricky since we need to match features back to original fragments
    """
    print("Reconstructing fragment pairs from dataset...")

    # We'll need to regenerate the pairs by matching features
    # Extract time_gap and spatial_gap (first two features) as identifiers
    pairs_info = []

    # Group fragments by GT ID for positive pairs
    from collections import defaultdict
    vehicle_fragments = defaultdict(list)
    for idx, frag in enumerate(all_fragments):
        gt_id = get_gt_id(frag)
        if gt_id is not None:
            vehicle_fragments[gt_id].append((idx, frag))

    # For each row in dataset, try to find matching pair
    for row_idx in tqdm(range(len(X_basic)), desc="Matching pairs"):
        time_gap = X_basic[row_idx, 0]
        spatial_gap = X_basic[row_idx, 4] if X_basic.shape[1] > 4 else X_basic[row_idx, 1]

        # Search through all possible pairs
        found = False
        for gt_id, frag_list in vehicle_fragments.items():
            for i in range(len(frag_list)):
                for j in range(i + 1, len(frag_list)):
                    idx_a, frag_a = frag_list[i]
                    idx_b, frag_b = frag_list[j]

                    # Check if this matches the row
                    computed_time_gap = frag_b['first_timestamp'] - frag_a['last_timestamp']
                    if frag_a['direction'] == 1:
                        computed_spatial_gap = frag_b['starting_x'] - frag_a['ending_x']
                    else:
                        computed_spatial_gap = frag_a['starting_x'] - frag_b['ending_x']

                    # Match with tolerance
                    if abs(computed_time_gap - time_gap) < 1e-3 and abs(computed_spatial_gap - spatial_gap) < 1e-3:
                        pairs_info.append((idx_a, idx_b))
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if not found:
            # This must be a negative pair - harder to match
            # For now, append placeholder
            pairs_info.append((None, None))

    return pairs_info


def add_advanced_features_simple():
    """
    Simplified approach: Regenerate dataset with advanced features directly
    """
    print("="*60)
    print("REGENERATING DATASET WITH ADVANCED FEATURES")
    print("="*60)

    # Load all fragment data
    all_X = []
    all_y = []
    all_scenarios = []
    all_mask_idx = []
    all_source_split_tag = []
    all_pair_id = []
    all_features_names = None

    for dataset_name in DATASETS:
        print(f"\nProcessing dataset: {dataset_name}")

        # Load data
        raw_path = DATASET_DIR / f"RAW_{dataset_name}.json"
        with open(raw_path, 'r') as f:
            raw_fragments = json.load(f)

        # Load existing dataset for this scene to get pairs
        # For simplicity, we'll just regenerate pairs here
        from enhanced_dataset_creation import DatasetGenerator

        generator = DatasetGenerator(max_time_gap=5.0, max_spatial_gap=200, y_threshold=5.0)

        # Generate positive pairs
        positive_pairs = generator.generate_positive_pairs(raw_fragments)
        print(f"  Positive pairs: {len(positive_pairs)}")

        # Generate negative pairs
        negative_pairs = generator.generate_hard_negatives(raw_fragments, num_negatives=len(positive_pairs))
        print(f"  Negative pairs: {len(negative_pairs)}")

        # Now add advanced features to each pair
        X_rows = []
        y_labels = []
        scenario_rows = []
        mask_idx_rows = []
        source_split_rows = []
        pair_id_rows = []

        all_pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

        print(f"  Extracting advanced features...")
        for (idx_a, idx_b, basic_features), label in tqdm(zip(all_pairs, labels), total=len(all_pairs)):
            frag_a = raw_fragments[idx_a]
            frag_b = raw_fragments[idx_b]

            # Get basic features
            feature_dict = basic_features.copy()

            # Add advanced features
            advanced_feats = extract_advanced_features(frag_a, frag_b)
            feature_dict.update(advanced_feats)

            if all_features_names is None:
                all_features_names = list(feature_dict.keys())

            X_rows.append([feature_dict[fname] for fname in all_features_names])
            y_labels.append(label)
            scenario_rows.append(dataset_name)
            mask_idx_rows.append(-1)
            source_split_rows.append(f"{dataset_name}_raw_pair")
            pair_type = "pos" if label == 1 else "neg"
            pair_id_rows.append(f"{dataset_name}:{pair_type}:{idx_a}:{idx_b}")

        X = np.array(X_rows)
        y = np.array(y_labels)

        print(f"  Dataset shape: {X.shape}")
        all_X.append(X)
        all_y.append(y)
        all_scenarios.append(np.array(scenario_rows, dtype=object))
        all_mask_idx.append(np.array(mask_idx_rows, dtype=np.int32))
        all_source_split_tag.append(np.array(source_split_rows, dtype=object))
        all_pair_id.append(np.array(pair_id_rows, dtype=object))

    # Combine all datasets
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    scenario_combined = np.hstack(all_scenarios)
    mask_idx_combined = np.hstack(all_mask_idx)
    source_split_combined = np.hstack(all_source_split_tag)
    pair_id_combined = np.hstack(all_pair_id)

    print("\n" + "="*60)
    print("COMBINED DATASET WITH ADVANCED FEATURES")
    print("="*60)
    print(f"Total pairs: {len(y_combined)}")
    print(f"Positive pairs: {np.sum(y_combined == 1)} ({np.sum(y_combined == 1) / len(y_combined) * 100:.2f}%)")
    print(f"Negative pairs: {np.sum(y_combined == 0)} ({np.sum(y_combined == 0) / len(y_combined) * 100:.2f}%)")
    print(f"Feature dimensions: {X_combined.shape[1]}")
    print(f"\nFeature names ({len(all_features_names)}):")
    for i, fname in enumerate(all_features_names):
        print(f"  {i+1:2d}. {fname}")

    # Save enhanced dataset
    output_file = WORKING_DIR / "training_dataset_advanced.npz"
    np.savez_compressed(
        output_file,
        X=X_combined,
        y=y_combined,
        feature_names=all_features_names,
        scenario=scenario_combined,
        mask_idx=mask_idx_combined,
        source_split_tag=source_split_combined,
        pair_id=pair_id_combined,
    )

    print(f"\nEnhanced dataset saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    return X_combined, y_combined, all_features_names


if __name__ == "__main__":
    X, y, feature_names = add_advanced_features_simple()
    print("\n" + "="*60)
    print("Advanced feature extraction complete!")
    print("="*60)
