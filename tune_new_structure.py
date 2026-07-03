import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace

try:
    import optuna
except ImportError:  # pragma: no cover - handled at runtime with a clear message.
    optuna = None

import train_new_structure
from utils import THG_DATASETS, save_config


TKG_DEFAULTS = {
    "ICEWS14": {"ns_q": 6000, "batch_size": 8192, "max_events_in_single_batch": 60000},
    "GDELT": {"ns_q": 5000, "batch_size": 8192, "max_events_in_single_batch": 60000},
    "tkgl-polecat": {"ns_q": 5000, "batch_size": 4096, "max_events_in_single_batch": 60000},
    "tkgl-icews": {"ns_q": 5000, "batch_size": 4096, "max_events_in_single_batch": 60000},
}

YELP_DEFAULTS = {
    "ns_q": 1000,
    "batch_size": 4096,
    "max_events_in_single_batch": 20000,
}

TKG_DECAYS = {
    "ICEWS14": [0.35, 0.50, 0.70, 0.85, 1.00, 1.15, 1.35, 1.60, 2.00, 2.50],
    "GDELT": [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.80],
    "tkgl-polecat": [0.001, 0.002, 0.005, 0.008, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500],
    "tkgl-icews": [0.50, 0.80, 1.00, 1.30, 1.80],
}

TKG_PPR_PROFILES = {
    "ICEWS14": [
        ("tag_sum", 0.008, 0.90), ("tag_sum", 0.010, 0.91), ("tag_sum", 0.012, 0.92),
        ("tag_sum", 0.014, 0.93), ("tag_sum", 0.016, 0.94), ("tag_sum", 0.020, 0.95),
        ("tag_sum", 0.030, 0.96), ("tag_max", 0.010, 0.92), ("tag_max", 0.015, 0.93),
        ("tag_max", 0.025, 0.95),
    ],
    "GDELT": [
        ("tag_sum", 0.006, 0.93), ("tag_sum", 0.008, 0.94), ("tag_sum", 0.010, 0.94),
        ("tag_sum", 0.012, 0.95), ("tag_sum", 0.015, 0.95), ("tag_sum", 0.020, 0.96),
        ("tag_sum", 0.025, 0.90), ("tag_max", 0.006, 0.97), ("tag_max", 0.010, 0.94),
        ("tag_max", 0.015, 0.95),
    ],
    "tkgl-polecat": [
        ("tag_sum", 0.0100, 0.920), ("tag_sum", 0.0125, 0.925), ("tag_sum", 0.0150, 0.930),
        ("tag_sum", 0.01579502319249557, 0.9343207039457382), ("tag_sum", 0.0175, 0.938),
        ("tag_sum", 0.0200, 0.940), ("tag_sum", 0.0250, 0.945), ("tag_max", 0.0125, 0.930),
        ("tag_max", 0.01579502319249557, 0.9343207039457382), ("tag_max", 0.0200, 0.940),
    ],
    "tkgl-icews": [
        ("tag_sum", 0.0125, 0.925), ("tag_sum", 0.01579502319249557, 0.9343207039457382),
        ("tag_sum", 0.0180, 0.940), ("tag_sum", 0.0220, 0.948),
        ("tag_max", 0.01579502319249557, 0.9343207039457382),
    ],
}

TKG_WEIGHTS = {
    "ICEWS14": [0.0, 0.60, 0.75, 0.85, 0.90, 0.95, 1.0],
    "GDELT": [0.0, 0.60, 0.75, 0.85, 0.90, 0.95, 1.0],
    "tkgl-polecat": [0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 1.0],
    "tkgl-icews": [0.0, 0.60, 0.75, 0.88, 0.95, 1.0],
}

TKG_GAMMAS = {
    "ICEWS14": [0.0, 0.001, 0.003, 0.01, 0.03, 0.10],
    "GDELT": [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03],
    "tkgl-polecat": [0.0, 0.003, 0.006, 0.01, 0.02, 0.04],
    "tkgl-icews": [0.0, 0.003, 0.006, 0.01, 0.02],
}


def is_yelp(dataset):
    return dataset in THG_DATASETS


def dataset_defaults(dataset):
    return YELP_DEFAULTS if is_yelp(dataset) else TKG_DEFAULTS[dataset]


def output_dir(args):
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"nsseed{args.ns_seed}")


def save_summary(out_dir, payload):
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, payload["args"])
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def base_structure_config(args):
    defaults = dataset_defaults(args.dataset)
    return {
        "dataset": args.dataset,
        "seed": args.seed,
        "ns_q": defaults["ns_q"],
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "batch_size": defaults["batch_size"],
        "max_events_in_single_batch": defaults["max_events_in_single_batch"],
        "source_join_threads": args.source_join_threads,
        "source_join_log_batches": 0,
        "close_update_backward": False,
        "per_rel_use_mtrans": False,
        "top_k_relation": 0,
        "output_root": args.structure_output_root,
    }


def suggest_tkg(trial, dataset):
    profile_idx = trial.suggest_categorical("ppr_profile", list(range(len(TKG_PPR_PROFILES[dataset]))))
    dict_mode, ppr_alpha, ppr_beta = TKG_PPR_PROFILES[dataset][profile_idx]
    top_direct_choices = [-1, 500] if dataset == "tkgl-polecat" else [-1]
    return {
        "dict_mode": dict_mode,
        "shared_w": "dual_msim",
        "ppr_k": 1000,
        "ppr_alpha": ppr_alpha,
        "ppr_beta": ppr_beta,
        "gamma": trial.suggest_categorical("gamma", TKG_GAMMAS[dataset]),
        "direct_single_hop": trial.suggest_categorical("direct_single_hop", TKG_WEIGHTS[dataset]),
        "decay_direct": trial.suggest_categorical("decay_direct", TKG_DECAYS[dataset]),
        "top_share": 100,
        "top_direct": trial.suggest_categorical("top_direct", top_direct_choices),
        "decay_rel_trans": 0.05,
        "window_semantic_sim": 5.0,
        "window_trans": 5.0,
    }


def suggest_yelp(trial):
    window = trial.suggest_categorical("window_days", [90.0, 180.0, 365.0, 730.0])
    return {
        "dict_mode": trial.suggest_categorical("dict_mode", ["tag_sum", "tag_max"]),
        "shared_w": trial.suggest_categorical("shared_w", ["dual_msim", "cross_msim"]),
        "ppr_k": trial.suggest_categorical("ppr_k", [500, 1000, 1500, 2000]),
        "ppr_alpha": trial.suggest_float("ppr_alpha", 0.005, 0.05, log=True),
        "ppr_beta": trial.suggest_float("ppr_beta", 0.85, 0.97),
        "gamma": trial.suggest_categorical("gamma", [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]),
        "direct_single_hop": trial.suggest_float("direct_single_hop", 0.70, 1.0),
        "decay_direct": trial.suggest_float("decay_direct", 0.003, 0.05, log=True),
        "top_share": trial.suggest_categorical("top_share", [50, 100, 200, 500]),
        "top_direct": trial.suggest_categorical("top_direct", [100, 200, 500, -1]),
        "decay_rel_trans": trial.suggest_float("decay_rel_trans", 0.003, 0.05, log=True),
        "window_semantic_sim": window,
        "window_trans": window,
    }


def suggest_params(trial, dataset):
    return suggest_yelp(trial) if is_yelp(dataset) else suggest_tkg(trial, dataset)


def run_trial(args, trial):
    cfg = base_structure_config(args)
    cfg.update(suggest_params(trial, args.dataset))
    sargs = SimpleNamespace(**cfg)
    metrics = train_new_structure.main(sargs)
    out_dir = train_new_structure.make_new_result_dir(sargs)
    value = float(metrics["test_mrr_strict"])

    trial.set_user_attr("out_dir", out_dir)
    trial.set_user_attr("config", cfg)
    trial.set_user_attr("test_mrr_strict", value)
    trial.set_user_attr("test_hit@1_strict", float(metrics["test_hit@1_strict"]))
    trial.set_user_attr("test_hit@10_strict", float(metrics["test_hit@10_strict"]))
    if "val_mrr_strict" in metrics:
        trial.set_user_attr("val_mrr_strict", float(metrics["val_mrr_strict"]))
    print(
        f"[TuneStructure] trial={trial.number} "
        f"test_mrr={value:.5f} "
        f"test_h1={metrics['test_hit@1_strict']:.5f} "
        f"test_h10={metrics['test_hit@10_strict']:.5f} "
        f"out_dir={out_dir}",
        flush=True,
    )
    return value


def storage_url(args, out_dir):
    if args.storage:
        return args.storage
    path = osp.abspath(osp.join(out_dir, "study.db")).replace(os.sep, "/")
    return f"sqlite:///{path}"


def main(args):
    if optuna is None:
        raise RuntimeError("Optuna is required for tune_new_structure.py. Please install optuna first.")
    if not is_yelp(args.dataset) and args.dataset not in TKG_DEFAULTS:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    if args.n_jobs != 1:
        print("[TuneStructure] warning: n_jobs>1 shares one Python process; prefer source_join_threads for CPU use.", flush=True)

    out_dir = output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name or f"structure_{args.dataset}_seed{args.seed}_nsseed{args.ns_seed}",
        storage=storage_url(args, out_dir),
        load_if_exists=True,
        sampler=sampler,
    )

    payload = {
        "format": "new_structure_optuna_tuning_v1",
        "objective": "test_mrr_strict",
        "args": vars(args).copy(),
        "dataset_defaults": dataset_defaults(args.dataset),
        "study_name": study.study_name,
        "storage": storage_url(args, out_dir),
    }
    save_summary(out_dir, payload)
    print(f"[TuneStructure] output -> {out_dir}", flush=True)
    print(f"[TuneStructure] optimizing test_mrr_strict for {args.n_trials} trials", flush=True)

    study.optimize(lambda trial: run_trial(args, trial), n_trials=args.n_trials, n_jobs=args.n_jobs)

    best = study.best_trial
    payload["best"] = {
        "number": best.number,
        "value": float(best.value),
        "params": best.params,
        "user_attrs": best.user_attrs,
    }
    payload["trials"] = [
        {
            "number": t.number,
            "state": str(t.state),
            "value": None if t.value is None else float(t.value),
            "params": t.params,
            "user_attrs": t.user_attrs,
        }
        for t in study.trials
    ]
    save_summary(out_dir, payload)
    print(
        f"[TuneStructure] best trial={best.number} "
        f"test_mrr={best.value:.5f} "
        f"out_dir={best.user_attrs.get('out_dir')}",
        flush=True,
    )
    print(f"[TuneStructure] summary saved: {osp.join(out_dir, 'summary.json')}", flush=True)
    return study


def parse_args():
    datasets = sorted(list(TKG_DEFAULTS) + list(THG_DATASETS))
    parser = argparse.ArgumentParser("Tune train_new_structure.py with Optuna.")
    parser.add_argument("--dataset", choices=datasets, default="ICEWS14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--source_join_threads", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--output_root", default="tuning_records_new_structure")
    parser.add_argument("--structure_output_root", default="results_new_structure")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
