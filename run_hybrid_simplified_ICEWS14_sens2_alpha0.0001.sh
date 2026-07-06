#!/usr/bin/env bash
set -euo pipefail

THREADS="${1:-60}"
mkdir -p logs-sens2

COMMON_ARGS=(--dataset ICEWS14 --seed 42 --ns_q 6000 --ns_seed 42 --train_predict_ratio 0.3 --time_dir results_time_tkg_single/ICEWS14/seed42/r9eb5b85515d8_topk30_mw5-15-30_ed96_hd192_bs4096_ebs384_neg6_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin --output_root results_hybrid_simplified_sens2 --component_output_root results_hybrid_simplified_components_sens2 --structure_output_root results_new_structure_sens2 --ablation none --hybrid_select_split test --query_batch_size 64 --rescue_topk 100 --rescue_min_pos_rank 1 --rescue_max_pos_rank 100 --num_threads "${THREADS}" --source_join_threads "${THREADS}" --n_estimators 64 --learning_rate 0.01 --num_leaves 63 --max_depth 12 --min_child_samples 200 --reg_lambda 3.0 --reg_alpha 0.1 --min_split_gain 0.01 --subsample 0.85 --colsample_bytree 0.75 --b_mode continuous --b_continuous_alpha 0.0001 --structure_batch_size 8192 --structure_max_events_in_single_batch 60000 --structure_dict_mode tag_sum --structure_shared_w dual_msim --structure_ppr_k 100 --structure_top_k_relation 0 --structure_ppr_alpha 0.0001 --structure_gamma 0.001 --structure_direct_single_hop 0.85 --structure_decay_direct 0.35 --structure_top_share 100 --structure_top_direct -1 --structure_decay_rel_trans 0.05 --structure_window_semantic_sim 5.0 --structure_window_trans 5.0)

for BETA in 0.6 0.7 0.8; do
  SID="icews14_sens2_alpha0.0001_beta${BETA}_k100"
  OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u train_hybrid_simplified.py "${COMMON_ARGS[@]}" --structure_id "${SID}" --component_dir "results_hybrid_simplified_components_sens2/ICEWS14/seed42/component_scores/${SID}" --structure_ppr_beta "${BETA}" > "logs-sens2/hybrid_simplified_ICEWS14_alpha0.0001_beta${BETA}_k100.log" 2>&1
done
