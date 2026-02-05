import motmetrics as mm
import numpy as np
import json
import os
from collections import defaultdict

# File paths
GT_FILE = './GT_i.json'
REC_FILE = './REC_i.json'
RAW_FILE = './RAW_i.json'

# Parameters
IOU_THRESHOLD = 0.3
FRAME_RATE = 25  # fps (0.04s per frame)


def load_trajectories(json_file):
    """Load I24 JSON trajectories."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def get_trajectory_id(traj):
    """Extract a unique integer ID from trajectory."""
    if '_id' in traj:
        _id = traj['_id']
        if isinstance(_id, dict) and '$oid' in _id:
            # Hash the MongoDB ObjectId string to integer
            return hash(_id['$oid']) % (2**31)
        elif isinstance(_id, str):
            return hash(_id) % (2**31)
        else:
            return int(_id)
    elif 'local_fragment_id' in traj:
        lfid = traj['local_fragment_id']
        if isinstance(lfid, list):
            return lfid[0]
        return lfid
    return 0


def get_length_width(traj, idx):
    """Get length and width at a specific index."""
    length = traj.get('length', 5.0)
    width = traj.get('width', 2.0)

    if isinstance(length, list):
        length = length[min(idx, len(length) - 1)]
    if isinstance(width, list):
        width = width[min(idx, len(width) - 1)]

    return float(length), float(width)


def get_time_range(trajectories):
    """Get min and max timestamps from trajectories."""
    all_timestamps = []
    for traj in trajectories:
        timestamps = traj.get('timestamp', [])
        if timestamps:
            all_timestamps.extend(timestamps)
    if not all_timestamps:
        return None, None
    return min(all_timestamps), max(all_timestamps)


def trajectories_to_frame_data(trajectories, frame_rate=25, ref_min_time=None):
    """
    Convert trajectory list to frame-indexed dict.
    Returns: {frame_num: [(id, bbox), ...]} where bbox = [x, y, length, width]
    """
    if not trajectories:
        return {}

    min_time, max_time = get_time_range(trajectories)
    if min_time is None:
        return {}

    # Use reference time if provided (for aligning multiple files)
    if ref_min_time is not None:
        min_time = ref_min_time

    frame_duration = 1.0 / frame_rate

    # Build frame-indexed data
    frame_data = defaultdict(list)

    for traj in trajectories:
        traj_id = get_trajectory_id(traj)
        timestamps = traj.get('timestamp', [])
        x_positions = traj.get('x_position', [])
        y_positions = traj.get('y_position', [])

        if not timestamps or not x_positions or not y_positions:
            continue

        for idx, (t, x, y) in enumerate(zip(timestamps, x_positions, y_positions)):
            frame_num = int((t - min_time) / frame_duration)
            length, width = get_length_width(traj, idx)

            # Bbox as [x_center, y_center, length, width] - will convert for IOU later
            bbox = [x, y, length, width]
            frame_data[frame_num].append((traj_id, bbox))

    return dict(frame_data)


def bbox_center_to_corner(bbox):
    """Convert [x_center, y_center, length, width] to [x_min, y_min, length, width]."""
    x, y, length, width = bbox
    return [x - length / 2, y - width / 2, length, width]


def compute_iou_matrix(gt_bboxes, t_bboxes, max_iou=0.3):
    """
    Compute IOU distance matrix (1 - IOU) between ground truth and tracker bboxes.
    Bboxes in format [x_min, y_min, length, width].
    Returns NaN for pairs with IOU < max_iou.
    """
    gt_bboxes = np.asarray(gt_bboxes, dtype=np.float64)
    t_bboxes = np.asarray(t_bboxes, dtype=np.float64)

    n_gt = len(gt_bboxes)
    n_t = len(t_bboxes)

    if n_gt == 0 or n_t == 0:
        return np.empty((n_gt, n_t))

    # Convert to [x1, y1, x2, y2] format
    gt_x1 = gt_bboxes[:, 0]
    gt_y1 = gt_bboxes[:, 1]
    gt_x2 = gt_bboxes[:, 0] + gt_bboxes[:, 2]
    gt_y2 = gt_bboxes[:, 1] + gt_bboxes[:, 3]

    t_x1 = t_bboxes[:, 0]
    t_y1 = t_bboxes[:, 1]
    t_x2 = t_bboxes[:, 0] + t_bboxes[:, 2]
    t_y2 = t_bboxes[:, 1] + t_bboxes[:, 3]

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
    t_area = t_bboxes[:, 2] * t_bboxes[:, 3]
    union_area = gt_area[:, None] + t_area[None, :] - inter_area

    # Compute IOU
    iou = inter_area / np.maximum(union_area, 1e-10)

    # Convert to distance (1 - IOU), set to NaN if IOU < max_iou
    dist = 1 - iou
    dist[iou < max_iou] = np.nan

    return dist


def compute_mot_metrics(gt_file, tracker_file, name, iou_threshold=0.3):
    """Compute MOT metrics comparing tracker output to ground truth."""

    # Load trajectories
    gt_trajs = load_trajectories(gt_file)
    tracker_trajs = load_trajectories(tracker_file)

    print(f"Loaded {len(gt_trajs)} GT trajectories, {len(tracker_trajs)} tracker trajectories")

    # Find common time base (use minimum of both min_times)
    gt_min_t, gt_max_t = get_time_range(gt_trajs)
    t_min_t, t_max_t = get_time_range(tracker_trajs)
    ref_min_time = min(gt_min_t, t_min_t)

    # Convert to frame data using common time base
    gt_frames = trajectories_to_frame_data(gt_trajs, FRAME_RATE, ref_min_time)
    tracker_frames = trajectories_to_frame_data(tracker_trajs, FRAME_RATE, ref_min_time)

    # Use overlapping time range
    all_frames = set(gt_frames.keys()) | set(tracker_frames.keys())
    if not all_frames:
        print(f"No frames found!")
        return None

    max_frame = max(all_frames)
    print(f"Processing {max_frame + 1} frames...")

    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Process each frame
    for frame in range(max_frame + 1):
        gt_dets = gt_frames.get(frame, [])
        t_dets = tracker_frames.get(frame, [])

        if not gt_dets and not t_dets:
            continue

        # Extract IDs and bboxes
        gt_ids = [d[0] for d in gt_dets]
        gt_bboxes = np.array([bbox_center_to_corner(d[1]) for d in gt_dets]) if gt_dets else np.empty((0, 4))

        t_ids = [d[0] for d in t_dets]
        t_bboxes = np.array([bbox_center_to_corner(d[1]) for d in t_dets]) if t_dets else np.empty((0, 4))

        # Compute IOU distance matrix (1 - IOU, with max_iou threshold)
        C = compute_iou_matrix(gt_bboxes, t_bboxes, max_iou=iou_threshold)

        # Update accumulator
        acc.update(gt_ids, t_ids, C)

    # Compute object-level FM and IDsw
    events = acc.events
    obj_switches = set()
    obj_fragments = set()

    for _, event in events.iterrows():
        gt_id = event['OId']
        if event['Type'] == 'SWITCH':
            obj_switches.add(gt_id)
        elif event['Type'] == 'TRANSFER':
            obj_fragments.add(gt_id)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['precision', 'recall', 'mota', 'motp', 'num_objects',
                 'mostly_tracked', 'partially_tracked', 'mostly_lost',
                 'num_false_positives', 'num_misses', 'num_switches'],
        name=name
    )

    # Add object-level metrics
    summary['obj_switches'] = len(obj_switches)
    summary['obj_fragments'] = len(obj_fragments)

    # Transform MOTP (1 - distance = IOU)
    summary['motp'] = (1 - summary['motp'])

    # Tracker trajectory count
    tracker_traj_count = len(tracker_trajs)
    summary['no_trajs'] = tracker_traj_count

    # Compute Fgmt/GT and Sw/GT
    num_gt_vehicles = len(gt_trajs)
    summary['fgmt_per_gt'] = tracker_traj_count / num_gt_vehicles if num_gt_vehicles > 0 else 0
    summary['sw_per_gt'] = summary['num_switches'].values[0] / num_gt_vehicles if num_gt_vehicles > 0 else 0

    # Render summary
    strsummary = mm.io.render_summary(
        summary,
        formatters={
            'mota': '{:.3f}'.format,
            'motp': '{:.3f}'.format,
            'recall': '{:.2f}'.format,
            'precision': '{:.2f}'.format,
            'no_trajs': '{:.0f}'.format,
            'fgmt_per_gt': '{:.2f}'.format,
            'sw_per_gt': '{:.2f}'.format
        },
        namemap={
            'recall': 'Rcll',
            'precision': 'Prcn',
            'num_objects': 'GT',
            'mostly_tracked': 'MT',
            'partially_tracked': 'PT',
            'mostly_lost': 'ML',
            'num_false_positives': 'FP',
            'num_misses': 'FN',
            'num_switches': 'IDsw',
            'obj_switches': 'Obj IDsw',
            'obj_fragments': 'Obj FM',
            'mota': 'MOTA',
            'motp': 'MOTP',
            'no_trajs': 'No. trajs',
            'fgmt_per_gt': 'Fgmt/GT',
            'sw_per_gt': 'Sw/GT'
        }
    )
    print(f"\nResults for {name}:\n{strsummary}")
    return summary


if __name__ == '__main__':
    import sys

    # Get suffix from command line (default: 'i')
    suffix = sys.argv[1] if len(sys.argv) > 1 else 'i'

    # Build file paths
    gt_file = f'./GT_{suffix}.json'
    raw_file = f'./RAW_{suffix}.json'
    rec_file = f'./REC_{suffix}.json'

    # Handle RAW_*_Bhat.json files if standard names don't exist
    raw_bhat_file = f'./RAW_{suffix}_Bhat.json'
    if not os.path.exists(raw_file) and os.path.exists(raw_bhat_file):
        raw_file = raw_bhat_file

    print("=" * 60)
    print(f"MOT Metrics for I24 Trajectories (suffix: {suffix})")
    print("=" * 60)

    # Evaluate RAW vs GT
    if os.path.exists(gt_file) and os.path.exists(raw_file):
        print(f"\n--- RAW_{suffix} vs GT_{suffix} ---")
        raw_summary = compute_mot_metrics(gt_file, raw_file, f'RAW_{suffix}', IOU_THRESHOLD)
    else:
        print(f"\nSkipping RAW: GT={os.path.exists(gt_file)}, RAW={os.path.exists(raw_file)}")

    # Evaluate REC vs GT (if REC exists)
    if os.path.exists(gt_file) and os.path.exists(rec_file):
        print(f"\n--- REC_{suffix} vs GT_{suffix} ---")
        rec_summary = compute_mot_metrics(gt_file, rec_file, f'REC_{suffix}', IOU_THRESHOLD)
    else:
        print(f"\nSkipping REC: REC_{suffix}.json not found")

    # Evaluate REC_LR vs GT (if REC_LR exists)
    rec_lr_file = f'./REC_{suffix}_LR.json'
    if os.path.exists(gt_file) and os.path.exists(rec_lr_file):
        print(f"\n--- REC_{suffix} (Logistic Regression) vs GT_{suffix} ---")
        rec_lr_summary = compute_mot_metrics(gt_file, rec_lr_file, f'REC_{suffix}_LR', IOU_THRESHOLD)
    else:
        print(f"\nSkipping REC_LR: REC_{suffix}_LR.json not found")

    # GT vs GT sanity check
    if os.path.exists(gt_file):
        print(f"\n--- GT_{suffix} vs GT_{suffix} (sanity check) ---")
        gt_summary = compute_mot_metrics(gt_file, gt_file, f'GT_{suffix}', IOU_THRESHOLD)
