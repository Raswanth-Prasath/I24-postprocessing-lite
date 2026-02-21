#!/usr/bin/env bash
# PINN threshold sweep with correct pipeline resources
# Usage: bash pinn_sweep.sh

set -e
source activate i24

command -v jq >/dev/null || { echo "jq not found"; exit 1; }
mkdir -p models/outputs/pinn_sweep_$(date +%Y%m%d)
OUTDIR="models/outputs/pinn_sweep_$(date +%Y%m%d)"

run_eval () {
  tag="$1"
  cfg="$2"
  tsec="${3:-600}"

  echo "========== Running ${tag} =========="
  rm -f "REC_i_${tag}.json" "REC_i_${tag}.json.bak"

  # Save config for reproducibility
  cp "${cfg}" "${OUTDIR}/parameters_${tag}.json"

  if command -v timeout >/dev/null; then
    timeout "${tsec}" python pp_lite.py i --config "${cfg}" --tag "${tag}" 2>&1 | tee "${OUTDIR}/${tag}.pp.log" || true
  else
    python pp_lite.py i --config "${cfg}" --tag "${tag}" 2>&1 | tee "${OUTDIR}/${tag}.pp.log" || true
  fi

  [ -f "REC_i_${tag}.json" ] || { echo "[${tag}] missing REC output"; return 0; }

  python diagnose_json.py "REC_i_${tag}.json" --fix || true
  python hota_trackeval.py --gt-file GT_i.json --tracker-file "REC_i_${tag}.json" --name "${tag}" | tee "${OUTDIR}/${tag}.trackeval.txt"
}

# Thresholds: coarse + fine around the 1.75-2.25 sweet spot
for th in 1.50 1.75 1.85 1.90 1.95 2.00 2.10 2.25 2.50 3.00; do
  tag="PINN_T${th/./p}"
  cfg="/tmp/parameters_${tag}.json"

  jq ".stitcher_args.stitch_thresh=${th}
      | .stitcher_args.master_stitch_thresh=${th}
      | .stitcher_args.weight_gain=1.0
      | .worker_size=64
      | .stitcher_timeout=300
      | .reconciliation_pool_timeout=300
      | .reconciliation_writer_timeout=300
      | .write_temp_timeout=300" parameters_PINN.json > "${cfg}"

  run_eval "${tag}" "${cfg}" 600
done

echo ""
echo "========== SUMMARY =========="
printf "%-15s %6s %6s %6s %6s %6s %6s %6s %6s\n" "Tag" "HOTA" "MOTA" "Prec" "Recall" "IDsw" "FP" "Sw/GT" "Trajs"
for f in "${OUTDIR}"/*.trackeval.txt; do
  tag=$(basename "$f" .trackeval.txt)
  hota=$(grep -oP 'HOTA\s+\K[\d.]+' "$f" || echo "N/A")
  mota=$(grep -oP 'MOTA\s+\K[\d.]+' "$f" || echo "N/A")
  prec=$(grep -oP 'Precision\s+\K[\d.]+' "$f" || echo "N/A")
  rec=$(grep -oP 'Recall\s+\K[\d.]+' "$f" || echo "N/A")
  idsw=$(grep -oP 'IDsw\s+\K[\d]+' "$f" || echo "N/A")
  fp=$(grep -oP 'FP\s+\K[\d]+' "$f" || echo "N/A")
  swgt=$(grep -oP 'Sw/GT\s+\K[\d.]+' "$f" || echo "N/A")
  trajs=$(grep -oP 'No\. trajs\s+\K[\d]+' "$f" || echo "N/A")
  printf "%-15s %6s %6s %6s %6s %6s %6s %6s %6s\n" "$tag" "$hota" "$mota" "$prec" "$rec" "$idsw" "$fp" "$swgt" "$trajs"
done
