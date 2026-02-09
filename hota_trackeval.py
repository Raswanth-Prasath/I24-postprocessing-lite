"""
HOTA/CLEAR/Identity evaluation using the TrackEval library.

Usage:
    python hota_trackeval.py i      # Scenario i (free-flow)
    python hota_trackeval.py ii     # Scenario ii (snowy)
    python hota_trackeval.py iii    # Scenario iii (congested)
"""

import numpy as np
import os
import sys

from trackeval.metrics import HOTA, CLEAR, Identity

from mot_i24 import (
    load_trajectories,
    get_trajectory_id,
    get_length_width,
    get_time_range,
    trajectories_to_frame_data,
    bbox_center_to_corner,
    FRAME_RATE,
)


def compute_iou_similarity(gt_bboxes, tracker_bboxes):
    """
    Compute raw IoU similarity matrix between GT and tracker bounding boxes.

    Args:
        gt_bboxes: (N, 4) array in [x_min, y_min, length, width] format
        tracker_bboxes: (M, 4) array in [x_min, y_min, length, width] format

    Returns:
        (N, M) numpy array of IoU values in [0, 1]
    """
    gt_bboxes = np.asarray(gt_bboxes, dtype=np.float64)
    tracker_bboxes = np.asarray(tracker_bboxes, dtype=np.float64)

    n_gt = len(gt_bboxes)
    n_t = len(tracker_bboxes)

    if n_gt == 0 or n_t == 0:
        return np.zeros((n_gt, n_t), dtype=np.float64)

    # Convert [x_min, y_min, L, W] to [x1, y1, x2, y2]
    gt_x1 = gt_bboxes[:, 0]
    gt_y1 = gt_bboxes[:, 1]
    gt_x2 = gt_bboxes[:, 0] + gt_bboxes[:, 2]
    gt_y2 = gt_bboxes[:, 1] + gt_bboxes[:, 3]

    t_x1 = tracker_bboxes[:, 0]
    t_y1 = tracker_bboxes[:, 1]
    t_x2 = tracker_bboxes[:, 0] + tracker_bboxes[:, 2]
    t_y2 = tracker_bboxes[:, 1] + tracker_bboxes[:, 3]

    # Compute intersection
    inter_x1 = np.maximum(gt_x1[:, None], t_x1[None, :])
    inter_y1 = np.maximum(gt_y1[:, None], t_y1[None, :])
    inter_x2 = np.minimum(gt_x2[:, None], t_x2[None, :])
    inter_y2 = np.minimum(gt_y2[:, None], t_y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Compute union
    gt_area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
    t_area = tracker_bboxes[:, 2] * tracker_bboxes[:, 3]
    union_area = gt_area[:, None] + t_area[None, :] - inter_area

    iou = inter_area / np.maximum(union_area, 1e-10)
    return iou


def prepare_trackeval_data(gt_frames, tracker_frames):
    """
    Convert frame-indexed data to TrackEval's eval_sequence() format.

    Args:
        gt_frames: {frame_num: [(id, bbox), ...]} for ground truth
        tracker_frames: {frame_num: [(id, bbox), ...]} for tracker

    Returns:
        dict with keys: num_timesteps, num_gt_ids, num_tracker_ids,
              num_gt_dets, num_tracker_dets, gt_ids, tracker_ids,
              similarity_scores
    """
    # Get sorted union of all frame numbers
    all_frames = sorted(set(gt_frames.keys()) | set(tracker_frames.keys()))
    num_timesteps = len(all_frames)

    # Collect all unique IDs and build contiguous 0-based remapping
    all_gt_ids = set()
    all_tracker_ids = set()
    for frame in all_frames:
        for det_id, _ in gt_frames.get(frame, []):
            all_gt_ids.add(det_id)
        for det_id, _ in tracker_frames.get(frame, []):
            all_tracker_ids.add(det_id)

    gt_id_map = {orig_id: new_id for new_id, orig_id in enumerate(sorted(all_gt_ids))}
    tracker_id_map = {orig_id: new_id for new_id, orig_id in enumerate(sorted(all_tracker_ids))}

    num_gt_ids = len(gt_id_map)
    num_tracker_ids = len(tracker_id_map)

    # Build per-timestep arrays
    gt_ids_per_t = []
    tracker_ids_per_t = []
    similarity_per_t = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for frame in all_frames:
        gt_dets = gt_frames.get(frame, [])
        t_dets = tracker_frames.get(frame, [])

        # Remap GT IDs and collect bboxes
        gt_ids_t = np.array([gt_id_map[d[0]] for d in gt_dets], dtype=int)
        gt_bboxes_t = np.array(
            [bbox_center_to_corner(d[1]) for d in gt_dets], dtype=np.float64
        ).reshape(-1, 4)

        # Remap tracker IDs and collect bboxes
        tracker_ids_t = np.array([tracker_id_map[d[0]] for d in t_dets], dtype=int)
        tracker_bboxes_t = np.array(
            [bbox_center_to_corner(d[1]) for d in t_dets], dtype=np.float64
        ).reshape(-1, 4)

        # Compute IoU similarity matrix
        sim = compute_iou_similarity(gt_bboxes_t, tracker_bboxes_t)

        gt_ids_per_t.append(gt_ids_t)
        tracker_ids_per_t.append(tracker_ids_t)
        similarity_per_t.append(sim)

        num_gt_dets += len(gt_ids_t)
        num_tracker_dets += len(tracker_ids_t)

    return {
        'num_timesteps': num_timesteps,
        'num_gt_ids': num_gt_ids,
        'num_tracker_ids': num_tracker_ids,
        'num_gt_dets': num_gt_dets,
        'num_tracker_dets': num_tracker_dets,
        'gt_ids': gt_ids_per_t,
        'tracker_ids': tracker_ids_per_t,
        'similarity_scores': similarity_per_t,
    }


def evaluate_with_trackeval(gt_file, tracker_file, name):
    """
    Evaluate tracker output against ground truth using TrackEval metrics.

    Args:
        gt_file: path to ground truth JSON
        tracker_file: path to tracker output JSON
        name: label for printing

    Returns:
        dict of metric results
    """
    # Load trajectories
    try:
        gt_trajs = load_trajectories(gt_file)
        tracker_trajs = load_trajectories(tracker_file)
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Tip: run 'python diagnose_json.py <file> --fix' to repair malformed JSON")
        return None
    print(f"Loaded {len(gt_trajs)} GT trajectories, {len(tracker_trajs)} tracker trajectories")

    # Align time bases
    gt_min_t, _ = get_time_range(gt_trajs)
    t_min_t, _ = get_time_range(tracker_trajs)
    ref_min_time = min(gt_min_t, t_min_t)

    # Convert to frame data
    gt_frames = trajectories_to_frame_data(gt_trajs, FRAME_RATE, ref_min_time)
    tracker_frames = trajectories_to_frame_data(tracker_trajs, FRAME_RATE, ref_min_time)

    if not gt_frames and not tracker_frames:
        print(f"No frames found!")
        return None

    # Prepare TrackEval data
    data = prepare_trackeval_data(gt_frames, tracker_frames)

    # Instantiate metrics (suppress CLEAR/Identity config printing)
    hota_metric = HOTA()
    clear_metric = CLEAR({'PRINT_CONFIG': False, 'THRESHOLD': 0.3})
    identity_metric = Identity({'PRINT_CONFIG': False})

    # Evaluate
    hota_res = hota_metric.eval_sequence(data)
    clear_res = clear_metric.eval_sequence(data)
    identity_res = identity_metric.eval_sequence(data)

    # Extract results
    # HOTA arrays are over 19 IoU thresholds - take mean
    hota_val = np.mean(hota_res['HOTA'])
    deta_val = np.mean(hota_res['DetA'])
    assa_val = np.mean(hota_res['AssA'])
    loca_val = np.mean(hota_res['LocA'])

    # CLEAR metrics (scalars)
    mota = clear_res['MOTA']
    motp = clear_res['MOTP']
    precision = clear_res['CLR_Pr']
    recall = clear_res['CLR_Re']
    idsw = clear_res['IDSW']
    fp = clear_res['CLR_FP']
    fn = clear_res['CLR_FN']
    frag = clear_res['Frag']

    # Identity metrics (scalars)
    idf1 = identity_res['IDF1']
    idr = identity_res['IDR']
    idp = identity_res['IDP']

    # Custom metrics
    num_gt_vehicles = len(gt_trajs)
    num_tracker_trajs = len(tracker_trajs)
    fgmt_per_gt = num_tracker_trajs / num_gt_vehicles if num_gt_vehicles > 0 else 0
    sw_per_gt = idsw / num_gt_vehicles if num_gt_vehicles > 0 else 0

    results = {
        'HOTA': hota_val,
        'DetA': deta_val,
        'AssA': assa_val,
        'LocA': loca_val,
        'MOTA': mota,
        'MOTP': motp,
        'IDF1': idf1,
        'IDR': idr,
        'IDP': idp,
        'Precision': precision,
        'Recall': recall,
        'IDsw': idsw,
        'FP': fp,
        'FN': fn,
        'Frag': frag,
        'No. trajs': num_tracker_trajs,
        'Fgmt/GT': fgmt_per_gt,
        'Sw/GT': sw_per_gt,
    }

    # Print formatted table
    print(f"\n{'=' * 50}")
    print(f"  Results for {name}")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<15} {'Value':>10}")
    print(f"  {'-' * 30}")
    fmt = [
        ('HOTA',      '{:.3f}'),
        ('DetA',      '{:.3f}'),
        ('AssA',      '{:.3f}'),
        ('LocA',      '{:.3f}'),
        ('MOTA',      '{:.3f}'),
        ('MOTP',      '{:.3f}'),
        ('IDF1',      '{:.3f}'),
        ('IDR',       '{:.3f}'),
        ('IDP',       '{:.3f}'),
        ('Precision',  '{:.3f}'),
        ('Recall',     '{:.3f}'),
        ('IDsw',      '{:.0f}'),
        ('FP',        '{:.0f}'),
        ('FN',        '{:.0f}'),
        ('Frag',      '{:.0f}'),
        ('No. trajs', '{:.0f}'),
        ('Fgmt/GT',   '{:.2f}'),
        ('Sw/GT',     '{:.2f}'),
    ]
    for key, f in fmt:
        print(f"  {key:<15} {f.format(results[key]):>10}")
    print(f"{'=' * 50}\n")

    return results


if __name__ == '__main__':
    suffix = sys.argv[1] if len(sys.argv) > 1 else 'i'

    script_dir = os.path.dirname(os.path.abspath(__file__))

    gt_file = os.path.join(script_dir, f'GT_{suffix}.json')
    raw_file = os.path.join(script_dir, f'RAW_{suffix}.json')
    rec_current_file = os.path.join(script_dir, f'REC_{suffix}.json')
    rec_lr_legacy_file = os.path.join(script_dir, f'REC_{suffix}_LR.json')

    def first_existing(*paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    rec_bm_file = first_existing(
        os.path.join(script_dir, f'REC_{suffix}_BM.json'),
        os.path.join(script_dir, f'REC_{suffix}_benchmark.json'),
        os.path.join(script_dir, f'REC_{suffix}_BM Copy.json'),
    )
    rec_bhat_file = first_existing(
        os.path.join(script_dir, f'REC_{suffix}_Bhat.json'),
        os.path.join(script_dir, f'REC_{suffix}_Bhat(old).json'),
    )
    rec_snn_file = first_existing(
        os.path.join(script_dir, f'REC_{suffix}_SNN.json'),
        os.path.join(script_dir, f'REC_{suffix}_siamese(old).json'),
    )

    # Handle renamed RAW files
    raw_bhat_file = os.path.join(script_dir, f'RAW_{suffix}_Bhat.json')
    if not os.path.exists(raw_file) and os.path.exists(raw_bhat_file):
        raw_file = raw_bhat_file

    print("=" * 60)
    print(f"TrackEval Metrics for I24 Trajectories (scenario: {suffix})")
    print("=" * 60)

    # RAW vs GT
    if os.path.exists(gt_file) and os.path.exists(raw_file):
        print(f"\n--- RAW_{suffix} vs GT_{suffix} ---")
        evaluate_with_trackeval(gt_file, raw_file, f'RAW_{suffix}')
    else:
        print(f"\nSkipping RAW: GT={os.path.exists(gt_file)}, RAW={os.path.exists(raw_file)}")

    # REC (current method) vs GT
    if os.path.exists(gt_file) and os.path.exists(rec_current_file):
        print(f"\n--- REC_{suffix} (current method) vs GT_{suffix} ---")
        evaluate_with_trackeval(gt_file, rec_current_file, f'REC_{suffix}')
    else:
        print(f"\nSkipping REC current: REC_{suffix}.json not found")

    # REC (Benchmark) vs GT
    if os.path.exists(gt_file) and rec_bm_file is not None:
        print(f"\n--- REC_{suffix} (Baseline) vs GT_{suffix} ---")
        print(f"Using baseline file: {os.path.basename(rec_bm_file)}")
        evaluate_with_trackeval(gt_file, rec_bm_file, f'REC_{suffix}_BM')
    else:
        print(f"\nSkipping REC baseline: no BM/benchmark file found")

    # REC (Bhat) vs GT
    if os.path.exists(gt_file) and rec_bhat_file is not None:
        print(f"\n--- REC_{suffix}_Bhat vs GT_{suffix} ---")
        print(f"Using Bhat file: {os.path.basename(rec_bhat_file)}")
        evaluate_with_trackeval(gt_file, rec_bhat_file, f'REC_{suffix}_Bhat')
    else:
        print(f"\nSkipping REC Bhat: no Bhat file found")

    # # REC_LR variants vs GT
    # for feature_count in [6, 7 ,8, 9, 10, 11, 15, 25, 26]:
    #     rec_lr_file = os.path.join(script_dir, f'REC_{suffix}_LR_{feature_count}.json')

    #     # Backward compatibility for older single-file naming.
    #     if feature_count == 9 and not os.path.exists(rec_lr_file) and os.path.exists(rec_lr_legacy_file):
    #         rec_lr_file = rec_lr_legacy_file

    #     if os.path.exists(gt_file) and os.path.exists(rec_lr_file):
    #         print(f"\n--- REC_{suffix} (Logistic Regression - {feature_count} features) vs GT_{suffix} ---")
    #         evaluate_with_trackeval(gt_file, rec_lr_file, f'REC_{suffix}_LR_{feature_count}')
    #     else:
    #         print(f"\nSkipping REC_LR: REC_{suffix}_LR_{feature_count}.json not found")

    
    # REC (SNN) vs GT
    if os.path.exists(gt_file) and rec_snn_file is not None:
        print(f"\n--- REC_{suffix}_SNN vs GT_{suffix} ---")
        print(f"Using SNN file: {os.path.basename(rec_snn_file)}")
        evaluate_with_trackeval(gt_file, rec_snn_file, f'REC_{suffix}_SNN')
    else:
        print(f"\nSkipping REC SNN: no SNN file found")
        

    # GT vs GT sanity check
    if os.path.exists(gt_file):
        print(f"\n--- GT_{suffix} vs GT_{suffix} (sanity check) ---")
        evaluate_with_trackeval(gt_file, gt_file, f'GT_{suffix}')
