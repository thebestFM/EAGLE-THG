import argparse
import json
import os
import os.path as osp

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

import train_new_hybrid_save_top10 as hybrid


TIME_ARG_MAP = {
    "batch_size": "time_batch_size",
    "eval_batch_size": "time_eval_batch_size",
    "eval_neg_chunk": "time_eval_neg_chunk",
    "max_eval_pairs": "time_max_eval_pairs",
    "stream_eval_batch_events": "time_stream_eval_batch_events",
    "eval_node_preload_chunk": "time_eval_node_preload_chunk",
    "max_eval_node_cache_mb": "time_max_eval_node_cache_mb",
    "preload_eval_nodes": "time_preload_eval_nodes",
    "dense_eval_node_cache": "time_dense_eval_node_cache",
    "cache_eval_source": "time_cache_eval_source",
    "quick_val_events": "time_quick_val_events",
    "quick_val_fraction": "time_quick_val_fraction",
    "num_epochs": "time_num_epochs",
    "patience": "time_patience",
    "selection_metric": "time_selection_metric",
    "lr": "time_lr",
    "weight_decay": "time_weight_decay",
    "train_num_neg": "time_train_num_neg",
    "stream_train_batch_events": "time_stream_train_batch_events",
    "hard_neg_ratio": "time_hard_neg_ratio",
    "train_sampler": "time_train_sampler",
    "train_group_matrix_mb": "time_train_group_matrix_mb",
    "train_loss": "time_train_loss",
    "rank_margin": "time_rank_margin",
    "temperature": "time_temperature",
    "grad_clip": "time_grad_clip",
    "tolerance": "time_tolerance",
    "curriculum_decay": "time_curriculum_decay",
    "curriculum_raw_age": "time_curriculum_raw_age",
    "topk": "time_topk",
    "multi_windows": "time_multi_windows",
    "time_dim": "time_time_dim",
    "rel_dim": "time_rel_dim",
    "node_dim": "time_node_dim",
    "event_dim": "time_event_dim",
    "hidden_dim": "time_hidden_dim",
    "num_layers": "time_num_layers",
    "dropout": "time_dropout",
    "time_min": "time_time_min",
    "token_expansion_factor": "time_token_expansion_factor",
    "channel_expansion_factor": "time_channel_expansion_factor",
    "use_single_layer": "time_use_single_layer",
    "predictor_mode": "time_predictor_mode",
    "event_encoder": "time_event_encoder",
    "transformer_heads": "time_transformer_heads",
    "transformer_ff_dim": "time_transformer_ff_dim",
    "use_cross_history": "time_use_cross_history",
    "cross_heads": "time_cross_heads",
    "use_neighbor_id": "time_use_neighbor_id",
    "use_abs_time": "time_use_abs_time",
    "abs_time_periods": "time_abs_time_periods",
    "abs_time_harmonics": "time_abs_time_harmonics",
    "abs_time_use_raw": "time_abs_time_use_raw",
    "use_query_gate": "time_use_query_gate",
    "query_gate_type": "time_query_gate_type",
    "use_rank_pos": "time_use_rank_pos",
    "use_node_geo": "time_use_node_geo",
    "thg_time_days": "time_thg_time_days",
    "user_center_half_life_days": "time_user_center_half_life_days",
    "no_retrain_on_train_prefix": "time_no_retrain_on_train_prefix",
    "use_amp": "time_use_amp",
    "allow_tf32": "time_allow_tf32",
}

STRUCTURE_ARG_MAP = {
    "batch_size": "structure_batch_size",
    "max_events_in_single_batch": "structure_max_events_in_single_batch",
    "source_join_threads": "structure_source_join_threads",
    "close_update_backward": "structure_close_update_backward",
    "dict_mode": "structure_dict_mode",
    "shared_w": "structure_shared_w",
    "per_rel_use_mtrans": "structure_per_rel_use_mtrans",
    "ppr_k": "structure_ppr_k",
    "top_k_relation": "structure_top_k_relation",
    "ppr_alpha": "structure_ppr_alpha",
    "ppr_beta": "structure_ppr_beta",
    "gamma": "structure_gamma",
    "direct_single_hop": "structure_direct_single_hop",
    "decay_direct": "structure_decay_direct",
    "top_share": "structure_top_share",
    "top_direct": "structure_top_direct",
    "decay_rel_trans": "structure_decay_rel_trans",
    "window_semantic_sim": "structure_window_semantic_sim",
    "window_trans": "structure_window_trans",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics, focus_metric):
    candidates = {
        "test_mrr": ("test_mrr_strict", "test_mrr", "mrr_strict"),
        "test_hr1": ("test_hit@1_strict", "test_hr1", "hit@1_strict"),
        "test_hr10": ("test_hit@10_strict", "test_hr10", "hit@10_strict"),
    }[focus_metric]
    for key in candidates:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"metrics missing {focus_metric}: tried {candidates}")


def scan_runs(root, dataset, seed, ns_seed, train_predict_ratio, focus_metric):
    base = osp.join(root, dataset, f"seed{seed}")
    runs = []
    if not osp.isdir(base):
        return runs
    for dirpath, _, filenames in os.walk(base):
        if "config.json" not in filenames or "metrics.json" not in filenames:
            continue
        config = load_json(osp.join(dirpath, "config.json"))
        if str(config.get("dataset")) != str(dataset):
            continue
        if int(config.get("seed", seed)) != int(seed):
            continue
        if int(config.get("ns_seed", ns_seed)) != int(ns_seed):
            continue
        if abs(float(config.get("train_predict_ratio", train_predict_ratio)) - float(train_predict_ratio)) > 1e-12:
            continue
        metrics = load_json(osp.join(dirpath, "metrics.json"))
        try:
            score = metric_value(metrics, focus_metric)
        except KeyError:
            continue
        runs.append({"dir": dirpath, "config": config, "metrics": metrics, "score": score})
    return sorted(runs, key=lambda x: x["score"], reverse=True)


def apply_config(hargs, config, mapping):
    for src, dst in mapping.items():
        if src in config:
            setattr(hargs, dst, config[src])


def suggest_hybrid_params(trial):
    return {
        "hybrid_train_topk": trial.suggest_categorical("hybrid_train_topk", [20, 50, 100, 200, 500, -1]),
        "lgbm_n_estimators": trial.suggest_categorical("lgbm_n_estimators", [1, 2, 4, 8, 16, 32, 64, 128]),
        "lgbm_learning_rate": trial.suggest_float("lgbm_learning_rate", 0.001, 0.08, log=True),
        "lgbm_num_leaves": trial.suggest_categorical("lgbm_num_leaves", [3, 7, 15, 31, 63, 127]),
        "lgbm_max_depth": trial.suggest_categorical("lgbm_max_depth", [2, 4, 6, 8, 10, 12, -1]),
        "lgbm_min_child_samples": trial.suggest_int("lgbm_min_child_samples", 20, 300, log=True),
        "lgbm_reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-3, 10.0, log=True),
        "lgbm_reg_alpha": trial.suggest_float("lgbm_reg_alpha", 1e-5, 1.0, log=True),
        "lgbm_min_split_gain": trial.suggest_float("lgbm_min_split_gain", 0.0, 0.05),
        "lgbm_subsample": trial.suggest_float("lgbm_subsample", 0.75, 1.0),
        "lgbm_colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.65, 1.0),
    }


def make_hybrid_args(args, time_run, structure_run, trial_params):
    hargs = hybrid.parse_args([])
    hargs.dataset = args.dataset
    hargs.seed = args.seed
    hargs.ns_seed = args.ns_seed
    hargs.train_predict_ratio = args.train_predict_ratio
    hargs.ns_q = int(time_run["config"].get("ns_q", args.ns_q))
    hargs.query_batch_size = args.query_batch_size
    hargs.time_root = args.time_root
    hargs.structure_output_root = args.structure_root
    hargs.output_root = args.hybrid_output_root
    hargs.num_threads = args.num_threads
    hargs.structure_source_join_threads = args.structure_source_join_threads
    hargs.max_train_queries = args.max_train_queries
    hargs.train_query_stride = args.train_query_stride
    apply_config(hargs, time_run["config"], TIME_ARG_MAP)
    apply_config(hargs, structure_run["config"], STRUCTURE_ARG_MAP)
    for key, value in trial_params.items():
        setattr(hargs, key, value)
    return hargs


def run_pair_study(args, pair_idx, time_run, structure_run, summary):
    study_name = f"{args.study_name}_pair{pair_idx}" if args.study_name else f"{args.dataset}_hybrid_pair{pair_idx}"
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed + int(pair_idx))
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)

    def objective(trial):
        params = suggest_hybrid_params(trial)
        hargs = make_hybrid_args(args, time_run, structure_run, params)
        metrics = hybrid.main(hargs)
        value = float(metrics["test_mrr_strict"])
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("time_dir", time_run["dir"])
        trial.set_user_attr("structure_dir", structure_run["dir"])
        trial.set_user_attr("hybrid_output_dir", metrics.get("model_path", ""))
        print(
            f"[TuneHybrid] pair={pair_idx} trial={trial.number} test_mrr={value:.5f} "
            f"time={time_run['score']:.5f} structure={structure_run['score']:.5f}",
            flush=True,
        )
        return value

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    record = {
        "pair_index": int(pair_idx),
        "time_dir": time_run["dir"],
        "time_score": float(time_run["score"]),
        "structure_dir": structure_run["dir"],
        "structure_score": float(structure_run["score"]),
        "best_trial": {
            "number": study.best_trial.number,
            "value": float(study.best_value),
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        },
    }
    summary["pairs"].append(record)
    save_summary(args, summary)
    return record


def output_dir(args):
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"focus={args.focus_metric}")


def save_summary(args, payload):
    out_dir = output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(args):
    if optuna is None:
        raise RuntimeError("Optuna is required for tune_hybrid.py")
    time_runs = scan_runs(
        args.time_root or "results_time_tkg_single",
        args.dataset,
        args.seed,
        args.ns_seed,
        args.train_predict_ratio,
        args.focus_metric,
    )[: int(args.top_time)]
    structure_runs = scan_runs(
        args.structure_root,
        args.dataset,
        args.seed,
        args.ns_seed,
        args.train_predict_ratio,
        args.focus_metric,
    )[: int(args.top_structure)]
    if not time_runs:
        raise RuntimeError("no matching time runs found")
    if not structure_runs:
        raise RuntimeError("no matching structure runs found")

    summary = {
        "format": "hybrid_optuna_god_view_v1",
        "args": vars(args).copy(),
        "top_time": time_runs,
        "top_structure": structure_runs,
        "pairs": [],
    }
    save_summary(args, summary)
    print(f"[TuneHybrid] top_time={len(time_runs)} top_structure={len(structure_runs)}", flush=True)
    pair_idx = 1
    for time_run in time_runs:
        for structure_run in structure_runs:
            run_pair_study(args, pair_idx, time_run, structure_run, summary)
            pair_idx += 1
    best = max(summary["pairs"], key=lambda r: float(r["best_trial"]["value"]))
    summary["best"] = best
    save_summary(args, summary)
    print(
        f"[TuneHybrid] best pair={best['pair_index']} test_mrr={best['best_trial']['value']:.5f}",
        flush=True,
    )
    print(f"[TuneHybrid] summary={osp.join(output_dir(args), 'summary.json')}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("Tune hybrid LightGBM parameters over top time/structure runs.")
    parser.add_argument("--dataset", default="Yelp-BOI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--focus_metric", choices=("test_mrr", "test_hr1", "test_hr10"), default="test_mrr")
    parser.add_argument("--top_time", type=int, default=3)
    parser.add_argument("--top_structure", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", default="")
    parser.add_argument("--time_root", default="results_time_tkg_single")
    parser.add_argument("--structure_root", default="results_new_structure")
    parser.add_argument("--hybrid_output_root", default="results_new_hybrid_save_top10")
    parser.add_argument("--output_root", default="tuning_records_hybrid")
    parser.add_argument("--query_batch_size", type=int, default=512)
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--structure_source_join_threads", type=int, default=1)
    parser.add_argument("--max_train_queries", type=int, default=0)
    parser.add_argument("--train_query_stride", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
