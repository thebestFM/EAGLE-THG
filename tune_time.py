import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

import train_time
from new_single_pipeline import time as time_module
from utils import THG_DATASETS, load_metrics


def base_time_args(args):
    values = {
        "dataset": args.dataset,
        "seed": args.seed,
        "gpu": args.gpu,
        "output_dir": "",
        "force": args.force,
        "batch_size": 1024,
        "eval_batch_size": 256,
        "eval_neg_chunk": 64,
        "max_eval_pairs": 62500,
        "stream_eval_batch_events": 16,
        "eval_node_preload_chunk": 65536,
        "max_eval_node_cache_mb": 4096.0,
        "preload_eval_nodes": True,
        "dense_eval_node_cache": True,
        "cache_eval_source": False,
        "eval_test": True,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "quick_val_events": 0,
        "quick_val_fraction": 0.2,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "selection_metric": "mrr",
        "lr": 8e-4,
        "weight_decay": 5e-5,
        "train_num_neg": 8,
        "stream_train_batch_events": 2048,
        "hard_neg_ratio": 0.5,
        "train_sampler": "grouped_exact",
        "train_group_matrix_mb": 512.0,
        "train_loss": "margin",
        "rank_margin": 1.0,
        "temperature": 1.0,
        "grad_clip": 1.0,
        "tolerance": 1e-8,
        "curriculum_decay": 0.0,
        "curriculum_raw_age": False,
        "topk": 40,
        "multi_windows": "10,40",
        "time_dim": 64,
        "rel_dim": 64,
        "node_dim": 64,
        "event_dim": 96,
        "hidden_dim": 160,
        "num_layers": 1,
        "dropout": 0.15,
        "time_min": 1.0,
        "token_expansion_factor": 0.5,
        "channel_expansion_factor": 4.0,
        "use_single_layer": False,
        "predictor_mode": "diag",
        "event_encoder": "mixer",
        "transformer_heads": 2,
        "transformer_ff_dim": None,
        "use_cross_history": False,
        "cross_heads": 2,
        "use_neighbor_id": True,
        "use_abs_time": True,
        "abs_time_periods": "7,30,180,365",
        "abs_time_harmonics": 1,
        "abs_time_use_raw": False,
        "use_query_gate": True,
        "query_gate_type": "channel",
        "use_rank_pos": True,
        "use_node_geo": None,
        "thg_time_days": None,
        "user_center_half_life_days": 365.0,
        "no_retrain_on_train_prefix": not bool(args.with_oof_retrain),
        "reuse_no_retrain_full": True,
        "profile_sync": False,
        "use_amp": True,
        "allow_tf32": True,
    }
    return train_time.normalize_args(SimpleNamespace(**values))


def suggest_params(trial):
    return {
        "topk": trial.suggest_categorical("topk", [50, 100, 200]),
        "multi_windows": trial.suggest_categorical(
            "multi_windows",
            ["10,40", "20,60", "10,30,80", "10,40,80"],
        ),
        "event_dim": trial.suggest_categorical("event_dim", [64, 96]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 160, 192]),
        "lr": trial.suggest_float("lr", 5e-4, 1.2e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.10, 0.25),
        "user_center_half_life_days": trial.suggest_categorical(
            "user_center_half_life_days",
            [180.0, 365.0, 730.0],
        ),
    }


def metric_value(metrics, focus_metric):
    key = {"test_mrr": "test_mrr", "test_hit10": "test_hit10"}[focus_metric]
    if key not in metrics:
        raise KeyError(f"missing {key} in time metrics")
    return float(metrics[key])


def run_trial(args, trial):
    targs = base_time_args(args)
    for key, value in suggest_params(trial).items():
        setattr(targs, key, value)
    train_time.main(targs)
    out_dir = time_module.get_out_dir(targs)
    metrics = load_metrics(out_dir)
    value = metric_value(metrics, args.focus_metric)
    trial.set_user_attr("out_dir", out_dir)
    trial.set_user_attr("config", vars(targs).copy())
    trial.set_user_attr("test_mrr", float(metrics.get("test_mrr", 0.0)))
    trial.set_user_attr("test_hit10", float(metrics.get("test_hit10", 0.0)))
    print(
        f"[TuneTime] trial={trial.number} "
        f"{args.focus_metric}={value:.5f} "
        f"test_mrr={float(metrics.get('test_mrr', 0.0)):.5f} "
        f"test_hit10={float(metrics.get('test_hit10', 0.0)):.5f} "
        f"out_dir={out_dir}",
        flush=True,
    )
    return value


def output_dir(args):
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"focus={args.focus_metric}")


def save_summary(args, study):
    out_dir = output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "format": "time_optuna_god_view_v1",
        "objective": args.focus_metric,
        "args": vars(args).copy(),
        "best": {
            "number": study.best_trial.number,
            "value": float(study.best_value),
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        },
        "trials": [
            {
                "number": t.number,
                "state": str(t.state),
                "value": None if t.value is None else float(t.value),
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[TuneTime] summary={osp.join(out_dir, 'summary.json')}", flush=True)


def main(args):
    if optuna is None:
        raise RuntimeError("Optuna is required for tune_time.py")
    if args.dataset not in THG_DATASETS:
        raise ValueError("tune_time.py currently supports Yelp/THG datasets only")
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name or f"time_{args.dataset}_seed{args.seed}_nsseed{args.ns_seed}",
        sampler=sampler,
    )
    print(
        f"[TuneTime] dataset={args.dataset} n_trials={args.n_trials} objective={args.focus_metric}",
        flush=True,
    )
    study.optimize(lambda trial: run_trial(args, trial), n_trials=args.n_trials, n_jobs=1)
    save_summary(args, study)
    print(
        f"[TuneTime] best trial={study.best_trial.number} "
        f"{args.focus_metric}={study.best_value:.5f} "
        f"out_dir={study.best_trial.user_attrs.get('out_dir')}",
        flush=True,
    )
    return study


def parse_args():
    parser = argparse.ArgumentParser("Tune train_time.py on Yelp datasets with Optuna.")
    parser.add_argument("--dataset", default="Yelp-BOI", choices=sorted(THG_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--focus_metric", choices=("test_mrr", "test_hit10"), default="test_mrr")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", default="")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--with_oof_retrain", action="store_true", default=False)
    parser.add_argument("--output_root", default="tuning_records_time")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
