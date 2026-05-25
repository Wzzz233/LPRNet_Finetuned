#!/bin/bash
# Batch re-evaluate all checkpoints for fair audit
# Usage: bash tools/routeA_nextstage/batch_fair_eval.sh

BASE="/home/wzzz/LPRNet"
SCRIPT="$BASE/tools/routeA_prime/eval_largecrop_and_fuse.py"
EVAL_DIR="$BASE/experiments/routeA_nextstage_20260512/fair_audit_evals"

mkdir -p "$EVAL_DIR"

declare -A CKPTS
CKPTS["B3_best"]="$BASE/experiments/routeA_prime_quadwarp_20260512/B3_fullplate_gray3_224x72_bal31/best.pt"
CKPTS["B3_last"]="$BASE/experiments/routeA_prime_quadwarp_20260512/B3_fullplate_gray3_224x72_bal31/last.pt"
CKPTS["G0_best"]="$BASE/experiments/routeA_nextstage_20260512/G0_baseline_repro/best.pt"
CKPTS["G0_last"]="$BASE/experiments/routeA_nextstage_20260512/G0_baseline_repro/last.pt"
CKPTS["G1_best"]="$BASE/experiments/routeA_nextstage_20260512/G1_boardlike_aug/best.pt"
CKPTS["G1_last"]="$BASE/experiments/routeA_nextstage_20260512/G1_boardlike_aug/last.pt"
CKPTS["G2_best"]="$BASE/experiments/routeA_nextstage_20260512/G2_real_upweight/best.pt"
CKPTS["G2_last"]="$BASE/experiments/routeA_nextstage_20260512/G2_real_upweight/last.pt"
CKPTS["G3_best"]="$BASE/experiments/routeA_nextstage_20260512/G3_boardlike_aug_plus_real/best.pt"
CKPTS["G3_last"]="$BASE/experiments/routeA_nextstage_20260512/G3_boardlike_aug_plus_real/last.pt"

for name in "${!CKPTS[@]}"; do
    ckpt="${CKPTS[$name]}"
    out_name="${name}_fair_audit"
    out_dir="$EVAL_DIR/$out_name"
    mkdir -p "$out_dir"

    echo "=== $name: $(basename $ckpt) ==="
    python "$SCRIPT" \
        --model "$ckpt" \
        --experiment_name "$out_name" \
        --input_size 224 72 --ocr_preproc gray3 --in_channels 1 2>&1 | tee "$out_dir/eval_output.log"

    # Copy results from routeA_prime dir to our fair_audit dir
    src_dir="$BASE/experiments/routeA_prime_quadwarp_20260512/$out_name"
    if [ -d "$src_dir" ]; then
        cp "$src_dir"/eval_*.json "$out_dir"/ 2>/dev/null
        rm -rf "$src_dir"
    fi
    echo ""
done

echo "=== ALL DONE ==="
