#!/usr/bin/env bash
set -euo pipefail

THREADS="${1:-60}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p logs_sens

for VALUE in 0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.5; do
  python -u train_new_structure.py --dataset ICEWS14 --seed 42 --ns_q 6000 --ns_seed 42 --train_predict_ratio 0.3 --output_root results_new_structure_sens --batch_size 8192 --max_events_in_single_batch 60000 --source_join_threads "$THREADS" --dict_mode tag_sum --shared_w dual_msim --ppr_k 1000 --top_k_relation 0 --ppr_alpha "$VALUE" --ppr_beta 0.96 --gamma 0.001 --direct_single_hop 0.85 --decay_direct 0.35 --top_share 100 --top_direct -1 --decay_rel_trans 0.05 --window_semantic_sim 5.0 --window_trans 5.0 > "logs_sens/structure_ICEWS14_alpha${VALUE}.log" 2>&1
done
