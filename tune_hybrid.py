import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

import train_new_hybrid_save_top10 as hybrid


METRIC_KEYS = {
    "test_mrr": ("test_mrr", "test_mrr_strict", "mrr_strict"),
    "test_hr1": ("test_hit@1_strict", "test_hr1", "hit@1_strict"),
    "test_hr10": ("test_hit@10_strict", "test_hit10", "test_hr10", "hit@10_strict"),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics, focus_metric):
    for key in METRIC_KEYS[focus_metric]:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"metrics missing {focus_metric}: tried {METRIC_KEYS[focus_metric]}")


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


def output_dir(args):
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"focus={args.focus_metric}")


def save_summary(args, payload):
    out_dir = output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def suggest_params(trial):
    rescue_topk = trial.suggest_categorical("rescue_topk", [50, 100, 200])
    return {
        "hybrid_preset_indices": str(trial.suggest_categorical("hybrid_preset", list(range(1, 10)))),
        "rescue_topk": int(rescue_topk),
        "rescue_min_pos_rank": 1,
        "rescue_max_pos_rank": int(rescue_topk),
        "b_mode": "continuous",
        "b_continuous_alpha": trial.suggest_float("b_continuous_alpha", 1e-5, 1e-2, log=True),
    }


def focus_arg(focus_metric):
    return {"test_mrr": "MRR", "test_hr1": "H1", "test_hr10": "H10"}[focus_metric]


def structure_value(config, key, default):
    value = config.get(key, default)
    return default if value is None else value


def make_hybrid_args(args, pair_idx, time_run, structure_run, trial_params):
    scfg = structure_run["config"]
    values = {
        "dataset": args.dataset,
        "seed": args.seed,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "best_hyper": "",
        "time_dir": time_run["dir"],
        "time_id": f"time_pair{pair_idx}",
        "time_root": "",
        "structure_dir": structure_run["dir"],
        "structure_id": f"structure_pair{pair_idx}",
        "structure_output_root": args.structure_root,
        "structure_batch_size": int(structure_value(scfg, "batch_size", 4096)),
        "structure_max_events_in_single_batch": int(structure_value(scfg, "max_events_in_single_batch", 20000)),
        "structure_close_update_backward": bool(structure_value(scfg, "close_update_backward", False)),
        "structure_dict_mode": str(structure_value(scfg, "dict_mode", "tag_sum")),
        "structure_shared_w": str(structure_value(scfg, "shared_w", "dual_msim")),
        "structure_per_rel_use_mtrans": bool(structure_value(scfg, "per_rel_use_mtrans", False)),
        "structure_ppr_k": int(structure_value(scfg, "ppr_k", 1000)),
        "structure_top_k_relation": int(structure_value(scfg, "top_k_relation", 0)),
        "structure_ppr_alpha": float(structure_value(scfg, "ppr_alpha", 0.01)),
        "structure_ppr_beta": float(structure_value(scfg, "ppr_beta", 0.9)),
        "structure_gamma": float(structure_value(scfg, "gamma", 0.0)),
        "structure_direct_single_hop": float(structure_value(scfg, "direct_single_hop", 1.0)),
        "structure_decay_direct": float(structure_value(scfg, "decay_direct", 0.01)),
        "structure_top_share": int(structure_value(scfg, "top_share", 100)),
        "structure_top_direct": int(structure_value(scfg, "top_direct", -1)),
        "structure_decay_rel_trans": float(structure_value(scfg, "decay_rel_trans", 0.01)),
        "structure_window_semantic_sim": float(structure_value(scfg, "window_semantic_sim", 365.0)),
        "structure_window_trans": float(structure_value(scfg, "window_trans", 365.0)),
        "output_root": args.hybrid_output_root,
        "query_batch_size": args.query_batch_size,
        "source_join_threads": args.source_join_threads,
        "source_join_log_batches": 0,
        "dsh_log_bucket_stats": False,
        "focus_metric": focus_arg(args.focus_metric),
        "hybrid_select_split": args.hybrid_select_split,
        "component_splits": "",
        "skip_time_score_check": False,
        "skip_component_metrics": args.skip_component_metrics,
        "skip_save_top10": args.skip_save_top10,
        "max_hybrid_train_queries": args.max_hybrid_train_queries,
        "hybrid_train_query_stride": args.hybrid_train_query_stride,
        "num_threads": args.num_threads,
        "b_binary_unseen": 0.0,
    }
    values.update(trial_params)
    return SimpleNamespace(**values)


def hybrid_score(summary, focus_metric):
    metrics = summary["best"]["test_metrics"]
    key = {"test_mrr": "mrr_strict", "test_hr1": "hit@1_strict", "test_hr10": "hit@10_strict"}[focus_metric]
    return float(metrics[key])


def run_pair_study(args, pair_idx, time_run, structure_run, summary):
    study_name = f"{args.study_name}_pair{pair_idx}" if args.study_name else f"{args.dataset}_rich_hybrid_pair{pair_idx}"
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed + int(pair_idx))
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)

    def objective(trial):
        hargs = make_hybrid_args(args, pair_idx, time_run, structure_run, suggest_params(trial))
        result = hybrid.run(hargs)
        value = hybrid_score(result, args.focus_metric)
        trial.set_user_attr("summary", result)
        trial.set_user_attr("time_dir", time_run["dir"])
        trial.set_user_attr("structure_dir", structure_run["dir"])
        print(
            f"[TuneHybrid] pair={pair_idx} trial={trial.number} "
            f"{args.focus_metric}={value:.5f} time={time_run['score']:.5f} "
            f"structure={structure_run['score']:.5f}",
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


def main(args):
    if optuna is None:
        raise RuntimeError("Optuna is required for tune_hybrid.py")
    if int(args.top_time) <= 0 or int(args.top_structure) <= 0:
        raise ValueError("--top_time and --top_structure must be > 0")
    if int(args.n_trials) <= 0:
        raise ValueError("--n_trials must be > 0")
    all_time_runs = scan_runs(
        args.time_root,
        args.dataset,
        args.seed,
        args.ns_seed,
        args.train_predict_ratio,
        args.focus_metric,
    )
    all_structure_runs = scan_runs(
        args.structure_root,
        args.dataset,
        args.seed,
        args.ns_seed,
        args.train_predict_ratio,
        args.focus_metric,
    )
    if len(all_time_runs) < int(args.top_time):
        raise RuntimeError(
            f"found {len(all_time_runs)} matching time runs under {args.time_root}, "
            f"but --top_time={args.top_time}"
        )
    if len(all_structure_runs) < int(args.top_structure):
        raise RuntimeError(
            f"found {len(all_structure_runs)} matching structure runs under {args.structure_root}, "
            f"but --top_structure={args.top_structure}"
        )
    time_runs = all_time_runs[: int(args.top_time)]
    structure_runs = all_structure_runs[: int(args.top_structure)]

    summary = {
        "format": "rich_rescue_hybrid_optuna_god_view_v1",
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
        f"[TuneHybrid] best pair={best['pair_index']} "
        f"{args.focus_metric}={best['best_trial']['value']:.5f}",
        flush=True,
    )
    print(f"[TuneHybrid] summary={osp.join(output_dir(args), 'summary.json')}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("Tune rich rescue hybrid over top time/structure runs.")
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
    parser.add_argument("--source_join_threads", type=int, default=1)
    parser.add_argument("--max_hybrid_train_queries", type=int, default=0)
    parser.add_argument("--hybrid_train_query_stride", type=int, default=1)
    parser.add_argument("--hybrid_select_split", choices=("val", "test"), default="test")
    parser.add_argument("--skip_component_metrics", action="store_true", default=False)
    parser.add_argument("--skip_save_top10", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
