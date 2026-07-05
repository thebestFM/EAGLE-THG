#!/usr/bin/env bash
set -euo pipefail

THREADS="${1:-60}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p logs_sens

for VALUE in 0.00001 0.0001 0.001 0.01 0.05 0.1 0.5 2; do
  python -u train_new_structure.py --dataset Yelp-BOI --seed 42 --ns_q 1000 --ns_seed 42 --train_predict_ratio 0.3 --output_root results_new_structure_sens --batch_size 4096 --max_events_in_single_batch 20000 --source_join_threads "$THREADS" --dict_mode tag_sum --shared_w cross_msim --ppr_k 2000 --top_k_relation 0 --ppr_alpha 0.04014299418112063 --ppr_beta 0.9138768591888399 --gamma "$VALUE" --direct_single_hop 0.9269715909367932 --decay_direct 0.00600910557188908 --top_share 100 --top_direct 500 --decay_rel_trans 0.00466069600031857 --window_semantic_sim 180.0 --window_trans 180.0 > "logs_sens/structure_Yelp-BOI_gamma${VALUE}.log" 2>&1
done
