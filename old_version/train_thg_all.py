import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace

from single_pipeline import thg_hybrid_single, thg_structure_single, thg_time_single


def common_params(cli):
    return {
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "ns_q": int(cli.ns_q),
        "ns_seed": int(cli.ns_seed),
        "train_predict_ratio": float(cli.train_predict_ratio),
    }


def make_time_args(cli, common):
    return SimpleNamespace(
        dataset=common["dataset"],
        seed=common["seed"],
        gpu=cli.gpu,
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        topk=cli.time_topk,
        node_dim=cli.time_node_dim,
        rel_dim=cli.time_rel_dim,
        hidden_dim=cli.time_hidden_dim,
        dropout=cli.time_dropout,
        batch_size=cli.time_batch_size,
        eval_batch_size=cli.time_eval_batch_size,
        eval_neg_chunk=cli.time_eval_neg_chunk,
        train_num_neg=cli.time_train_num_neg,
        hard_neg_ratio=cli.time_hard_neg_ratio,
        train_loss=cli.time_train_loss,
        temperature=cli.time_temperature,
        rank_margin=cli.time_rank_margin,
        lr=cli.time_lr,
        weight_decay=cli.time_weight_decay,
        grad_clip=cli.time_grad_clip,
        num_epochs=cli.time_num_epochs,
        patience=cli.time_patience,
        tolerance=cli.time_tolerance,
        selection_metric=cli.time_selection_metric,
        force=cli.force_time,
        eval_test=True,
    )


def make_structure_args(cli, common):
    return SimpleNamespace(
        dataset=common["dataset"],
        seed=common["seed"],
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        block_size=cli.structure_block_size,
        train_topk=cli.structure_train_topk,
        geo_topk=cli.geo_topk,
        geo_tau_km=cli.geo_tau_km,
        user_half_life_days=cli.user_half_life_days,
        business_half_life_days=cli.business_half_life_days,
        direct_weight=cli.direct_weight,
        geo_dynamic_weight=cli.geo_dynamic_weight,
        val_early_stop_tailk=cli.structure_val_early_stop_tailk,
        n_estimators=cli.lgbm_n_estimators,
        learning_rate=cli.lgbm_learning_rate,
        num_leaves=cli.lgbm_num_leaves,
        max_depth=cli.lgbm_max_depth,
        min_child_samples=cli.lgbm_min_child_samples,
        reg_lambda=cli.lgbm_reg_lambda,
        reg_alpha=cli.lgbm_reg_alpha,
        subsample=cli.lgbm_subsample,
        colsample_bytree=cli.lgbm_colsample_bytree,
        early_stopping_rounds=cli.lgbm_early_stopping_rounds,
        num_threads=cli.num_threads,
        force=cli.force_structure,
        force_feature_cache=cli.force_feature_cache,
        eval_test=True,
    )


def make_hybrid_args(cli, common, structure_dir, time_dir):
    return SimpleNamespace(
        dataset=common["dataset"],
        seed=common["seed"],
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        structure_dir=structure_dir,
        time_dir=time_dir,
        train_topk=cli.hybrid_train_topk,
        val_early_stop_tailk=cli.hybrid_val_early_stop_tailk,
        n_estimators=cli.hybrid_n_estimators,
        learning_rate=cli.hybrid_learning_rate,
        num_leaves=cli.hybrid_num_leaves,
        max_depth=cli.hybrid_max_depth,
        min_child_samples=cli.hybrid_min_child_samples,
        reg_lambda=cli.hybrid_reg_lambda,
        reg_alpha=cli.hybrid_reg_alpha,
        subsample=cli.hybrid_subsample,
        colsample_bytree=cli.hybrid_colsample_bytree,
        early_stopping_rounds=cli.hybrid_early_stopping_rounds,
        num_threads=cli.num_threads,
        force=cli.force_hybrid,
        force_feature_cache=cli.force_feature_cache,
        eval_test=True,
    )


def protocol_dir(common):
    return f"nq{common['ns_q']}_ns{common['ns_seed']}_tpr{common['train_predict_ratio']:g}"


def save_summary(common, summary):
    out_dir = osp.join("results_thg_all", common["dataset"], f"seed{common['seed']}", protocol_dir(common))
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[THG-All] saved summary -> {path}", flush=True)


def run_pipeline(cli):
    common = common_params(cli)
    time_args = make_time_args(cli, common)
    structure_args = make_structure_args(cli, common)
    time_dir = thg_time_single.get_out_dir(time_args)
    structure_dir = thg_structure_single.get_out_dir(structure_args)
    print(f"[THG-All] time -> {time_dir}", flush=True)
    time_metrics = thg_time_single.main(time_args)
    print(f"[THG-All] structure -> {structure_dir}", flush=True)
    structure_metrics = thg_structure_single.main(structure_args)
    hybrid_args = make_hybrid_args(cli, common, structure_dir, time_dir)
    hybrid_dir = thg_hybrid_single.get_out_dir(hybrid_args)
    print(f"[THG-All] hybrid -> {hybrid_dir}", flush=True)
    hybrid_metrics = thg_hybrid_single.main(hybrid_args)
    summary = {
        "format": "thg_pipeline_v1",
        **common,
        "time_dir": time_dir,
        "structure_dir": structure_dir,
        "hybrid_dir": hybrid_dir,
        "time_test_mrr": time_metrics.get("test_mrr_strict", time_metrics.get("test_mrr")),
        "structure_test_mrr": structure_metrics.get("test_mrr_strict", structure_metrics.get("test_mrr")),
        "hybrid_test_mrr": hybrid_metrics.get("test_mrr_strict", hybrid_metrics.get("test_mrr")),
        "time_metrics": time_metrics,
        "structure_metrics": structure_metrics,
        "hybrid_metrics": hybrid_metrics,
    }
    save_summary(common, summary)
    return summary


def load_args():
    parser = argparse.ArgumentParser("Run the THG Yelp time/structure/hybrid pipeline.")
    parser.add_argument("--dataset", type=str, default="Yelp-NOLA", choices=("Yelp-NOLA", "Yelp-PHL", "Yelp-TPA"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=8)

    parser.add_argument("--time_topk", type=int, default=40)
    parser.add_argument("--time_node_dim", type=int, default=64)
    parser.add_argument("--time_rel_dim", type=int, default=32)
    parser.add_argument("--time_hidden_dim", type=int, default=160)
    parser.add_argument("--time_dropout", type=float, default=0.15)
    parser.add_argument("--time_batch_size", type=int, default=1024)
    parser.add_argument("--time_eval_batch_size", type=int, default=256)
    parser.add_argument("--time_eval_neg_chunk", type=int, default=512)
    parser.add_argument("--time_train_num_neg", type=int, default=8)
    parser.add_argument("--time_hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--time_train_loss", type=str, default="margin", choices=("margin", "ce"))
    parser.add_argument("--time_temperature", type=float, default=1.0)
    parser.add_argument("--time_rank_margin", type=float, default=1.0)
    parser.add_argument("--time_lr", type=float, default=1e-3)
    parser.add_argument("--time_weight_decay", type=float, default=1e-5)
    parser.add_argument("--time_grad_clip", type=float, default=1.0)
    parser.add_argument("--time_num_epochs", type=int, default=20)
    parser.add_argument("--time_patience", type=int, default=4)
    parser.add_argument("--time_tolerance", type=float, default=1e-8)
    parser.add_argument("--time_selection_metric", type=str, default="hit10")

    parser.add_argument("--structure_block_size", type=int, default=128)
    parser.add_argument("--structure_train_topk", type=int, default=100)
    parser.add_argument("--geo_topk", type=int, default=30)
    parser.add_argument("--geo_tau_km", type=float, default=1.0)
    parser.add_argument("--user_half_life_days", type=float, default=365.0)
    parser.add_argument("--business_half_life_days", type=float, default=365.0)
    parser.add_argument("--direct_weight", type=float, default=1.0)
    parser.add_argument("--geo_dynamic_weight", type=float, default=0.25)
    parser.add_argument("--structure_val_early_stop_tailk", type=int, default=8000)

    parser.add_argument("--lgbm_n_estimators", type=int, default=800)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.04)
    parser.add_argument("--lgbm_num_leaves", type=int, default=63)
    parser.add_argument("--lgbm_max_depth", type=int, default=-1)
    parser.add_argument("--lgbm_min_child_samples", type=int, default=50)
    parser.add_argument("--lgbm_reg_lambda", type=float, default=1.0)
    parser.add_argument("--lgbm_reg_alpha", type=float, default=0.0)
    parser.add_argument("--lgbm_subsample", type=float, default=0.9)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--lgbm_early_stopping_rounds", type=int, default=50)

    parser.add_argument("--hybrid_train_topk", type=int, default=120)
    parser.add_argument("--hybrid_val_early_stop_tailk", type=int, default=5000)
    parser.add_argument("--hybrid_n_estimators", type=int, default=1000)
    parser.add_argument("--hybrid_learning_rate", type=float, default=0.03)
    parser.add_argument("--hybrid_num_leaves", type=int, default=63)
    parser.add_argument("--hybrid_max_depth", type=int, default=-1)
    parser.add_argument("--hybrid_min_child_samples", type=int, default=50)
    parser.add_argument("--hybrid_reg_lambda", type=float, default=1.0)
    parser.add_argument("--hybrid_reg_alpha", type=float, default=0.0)
    parser.add_argument("--hybrid_subsample", type=float, default=0.9)
    parser.add_argument("--hybrid_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--hybrid_early_stopping_rounds", type=int, default=50)

    parser.add_argument("--force_time", action="store_true", default=False)
    parser.add_argument("--force_structure", action="store_true", default=False)
    parser.add_argument("--force_hybrid", action="store_true", default=False)
    parser.add_argument("--force_feature_cache", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(load_args())
