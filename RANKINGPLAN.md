# Ranking-Loss Integration Plan (Revised and Decision-Complete)

  ## Summary

  Implement ranking-based training and integration in 4 phases, with explicit inference-path behavior, locked training hyperparameters, concrete split keys, mining
  diagnostics, and mandatory data-yield gates before full training.

  Locked strategy:

  1. Build anchor-candidate groups from pipeline replay.
  2. Train with weighted pairwise margin loss (GT primary + Bhattacharyya soft, 0.8/0.2).
  3. Add online semi-hard mining.
  4. Promote only if offline ranking gates and scenario-i downstream sanity gate pass.

  ———

  ## 1) Inference Path (Explicit Contract)

  ### Cost contract

  utils/stitch_cost_interface.py transformer path must output:

  - finite float
  - non-negative
  - lower is better
  - comparable to stitch threshold scale

  ### Ranking checkpoint runtime behavior

  Add transformer cost config fields:

  - training_objective: "classification" | "ranking"
  - score_mapping: "legacy_similarity" | "direct_cost"
  - calibration_mode: "off" | "linear" | "isotonic" | "quantile_match"

  Runtime mapping:

  1. If training_objective="classification": keep current path (similarity -> dissimilarity mapping -> scale_factor -> time_penalty -> calibration).
  2. If training_objective="ranking" and score_mapping="direct_cost":

  - model output is raw score
  - convert to base cost with softplus(raw_score) (stable non-negative)
  - total cost = base_cost + time_penalty * gap
  - apply calibration only if calibration_mode != "off".

  Calibration policy:

  - Default for ranking: calibration_mode="off" initially.
  - Enable isotonic only when threshold alignment is needed for pipeline sweeps.
  - Calibration artifact always maps raw_total -> bhat_total when enabled.

  Failure behavior:

  - invalid pair or NaN/Inf => return 1e6 (existing sentinel).
  - log counter metrics: invalid_cost_count, calibration_applied_count.

  ———

  ## 2) Dataset Build (Pipeline Replay)

  ### Source

  Reuse collect_pipeline_pairs_for_calibration() from models/evaluate_transformer.py.

  ### Output rows

  Each row:

  - scenario
  - anchor_id
  - candidate_id
  - track_anchor (track2)
  - track_candidate (track1)
  - gap
  - bhat_cost
  - gt_label (1/0/null)
  - direction
  - compute_node_id
  - group_size

  ### Group filters

  - keep anchors with 2 <= group_size <= 10
  - keep finite bhat_cost < 1e5
  - keep gt_label=null rows for soft term only

  ### Mandatory pre-commit yield validation

  Run replay on i/ii/iii and log:

  - total anchors
  - anchors passing 2–10
  - anchors with any GT
  - anchors with both positive and negative GT
  - class balance per anchor

  Abort training setup if:

  - total valid anchors < 200, or
  - anchors with mixed GT labels < 80.

  ———

  ## 3) Split Policy (Concrete Key)

  ### Primary split key

  Vehicle-disjoint split by GT vehicle ID from _get_gt_id():

  - priority: gt_ids[0][0].$oid
  - fallback: _source_gt_id

  ### Split behavior

  - anchors/candidates with GT IDs participate in vehicle-disjoint split.
  - rows with missing GT (gt_label=null) are train-only soft pool.
  - validation/test metrics are computed only on anchors with known GT and both classes present.

  This preserves leakage safety and keeps soft data usable without contaminating validation.

  ———

  ## 4) Loss and Training (Locked Hyperparameters)

  ### Pairwise loss

  Per anchor:

  - L_gt = mean(max(0, margin + s_pos - s_neg)) over GT cross-class pairs.
  - L_soft from Bhattacharyya ordering within same-class or missing-GT rows.
  - Total: L = 0.8 * L_gt + 0.2 * L_soft.
  - margin = 0.2.
  - per-anchor normalization, then batch mean.

  Soft weights:

  - w_soft = 0.3 * clamp(|bhat_i - bhat_j| / 2.0, 0.25, 1.0).

  ### Locked optimizer schedule

  - Optimizer: AdamW
  - lr = 1e-4
  - weight_decay = 1e-5
  - Scheduler: CosineAnnealingLR(T_max=30)
  - Max epochs: 30
  - Early stopping: monitor val_anchor_top1, patience 5
  - Gradient clipping: 1.0
  - Checkpoint: save best by val_anchor_top1

  Batching:

  - anchors_per_batch = 12
  - candidates_per_anchor = 6 (pad + mask)

  ———

  ## 5) Semi-Hard Mining (Online)

  Enable mining from epoch 3 (epochs 1-2 use full GT pairs).

  Per positive candidate:

  - choose negatives where 0 < (s_neg - s_pos) < margin + 0.3 (semi-hard)
  - include inversions (s_neg - s_pos) <= 0 (hard)
  - fallback random negative if none found
  - cap k=3 negatives per positive

  Add mining diagnostics:

  - semi_hard_rate
  - hard_rate
  - random_fallback_rate

  Gate:

  - if random_fallback_rate > 40% for 3 consecutive epochs, increase candidates_per_anchor to 8 for next run.

  ———

  ## 6) Evaluation and Promotion Gates

  ## Offline gates (held-out anchors)

  All required:

  1. GT cross-class pairwise accuracy >= 95%
  2. Anchor Top-1 accuracy >= 85%
  3. Mean per-anchor Spearman >= 0.55

  ## Downstream sanity gate

  Before full matrix:

  - run scenario i pipeline
  - require Sw/GT < 0.7

  If fail: stop promotion and investigate transfer gap.

  ## Full downstream evaluation

  Run i/ii/iii, compare vs Bhattacharyya and LR baselines:

  - Sw/GT, Fgmt/GT, MOTA, IDF1, HOTA.

  ———

  ## 7) File-Level Implementation Plan

  1. New: models/build_ranking_dataset.py

  - replay -> group -> filter -> split manifest -> yield report

  2. New: models/train_transformer_ranking.py

  - dataloader with anchor-group batching
  - weighted pairwise loss
  - mining
  - checkpointing + early stopping

  3. Update: models/transformer_model.py

  - objective-aware head/output path (classification vs ranking)

  4. Update: utils/stitch_cost_interface.py

  - ranking inference path (direct_cost)
  - calibration off mode
  - runtime counters/logging

  5. Update: models/evaluate_transformer.py

  - offline ranking metric report + promotion decision artifact

  6. Update configs:
  ———

  ## 8) Tests

  Unit:

  - GT extraction and split assignment
  - group filtering and yield counters
  - pairwise loss sign/margin correctness
  - mining selection + fallback accounting
  - inference mapping for ranking (softplus, calibration off/on)

  Integration:

  - 1-epoch smoke train on subset
  - evaluation artifact contains all gate metrics
  - scenario i pipeline run with ranking checkpoint writes valid REC output

  Regression:

  - classification path unchanged and still runnable

  ———

  ## Assumptions and Defaults

  - GT IDs are available for a meaningful subset; missing-GT rows are train-soft only.
  - Ranking model outputs unbounded scalar; softplus used for non-negativity in pipeline cost.
  - calibration_mode="off" by default for ranking checkpoints; isotonic only when needed for threshold alignment.
  - If pre-commit yield thresholds fail, broaden group filter to 2–12 before changing loss/mining.