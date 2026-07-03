import argparse
import ast
import copy
import json
import os.path as osp
from types import SimpleNamespace

from new_single_pipeline import time


HYBRID_CONFIG = osp.join("configs", "new_hybrid_inputs.json")

YELP_DATASETS = [
    "Yelp-NOLA",
    "Yelp-PHL",
    "Yelp-TPA",
    "Yelp-BOI",
    "Yelp-STL",
    "Yelp-SBA",
    "Yelp-RNO",
    "Yelp-IND",
    "Yelp-TUS",
    "Yelp-BNA",
]


DATASET_DEFAULTS = {
    "ICEWS14": {
        "ns_q": 6000,
        "batch_size": 4096,
        "eval_batch_size": 384,
        "eval_neg_chunk": 6000,
        "max_eval_pairs": 1800000,
        "train_group_matrix_mb": 2048.0,
        "quick_val_fraction": 0.3,
        "num_epochs": 50,
        "patience": 5,
    },
    "GDELT": {
        "ns_q": 5000,
        "batch_size": 2048,
        "eval_batch_size": 192,
        "eval_neg_chunk": 5000,
        "max_eval_pairs": 600000,
        "train_group_matrix_mb": 1024.0,
        "quick_val_fraction": 0.2,
        "num_epochs": 50,
        "patience": 5,
    },
    "tkgl-polecat": {
        "ns_q": 5000,
        "batch_size": 2048,
        "eval_batch_size": 128,
        "eval_neg_chunk": 5000,
        "max_eval_pairs": 400000,
        "train_group_matrix_mb": 512.0,
        "quick_val_fraction": 0.05,
        "num_epochs": 50,
        "patience": 3,
    },
    "tkgl-icews": {
        "ns_q": 5000,
        "batch_size": 2048,
        "eval_batch_size": 128,
        "eval_neg_chunk": 5000,
        "max_eval_pairs": 400000,
        "train_group_matrix_mb": 512.0,
        "quick_val_fraction": 0.05,
        "num_epochs": 50,
        "patience": 3,
    },
}

YELP_DEFAULTS = {
    "ns_q": 1000,
    "batch_size": 1024,
    "eval_batch_size": 256,
    "eval_neg_chunk": 512,
    "max_eval_pairs": 500000,
    "train_group_matrix_mb": 512.0,
    "quick_val_fraction": 0.2,
    "num_epochs": 20,
    "patience": 4,
}
for _dataset in YELP_DATASETS:
    DATASET_DEFAULTS[_dataset] = dict(YELP_DEFAULTS)


TIME_PRESETS = {
    "ICEWS14": {
        "time_cfg2_mrr": {
            "topk": 30,
            "multi_windows": "5,15,30",
            "train_num_neg": 6,
            "selection_metric": "mrr",
            "event_dim": 96,
            "hidden_dim": 192,
            "dropout": 0.1,
            "use_query_gate": True,
            "query_gate_type": "channel",
        },
        "time_cfg1_mrr": {
            "topk": 40,
            "multi_windows": "5,15,30,60",
            "train_num_neg": 8,
            "selection_metric": "mrr",
            "event_dim": 96,
            "hidden_dim": 192,
            "dropout": 0.1,
            "use_query_gate": True,
            "query_gate_type": "channel",
        },
        "time_cfg3_hr10": {
            "topk": 70,
            "multi_windows": "5,15,30,60,120",
            "train_num_neg": 4,
            "selection_metric": "hit10",
            "event_dim": 96,
            "hidden_dim": 192,
            "dropout": 0.1,
            "use_query_gate": True,
            "query_gate_type": "channel",
        },
    },
    "GDELT": {
        "time_mrr": {
            "topk": 60,
            "multi_windows": "7,30",
            "train_num_neg": 4,
            "selection_metric": "mrr",
            "event_dim": 64,
            "hidden_dim": 128,
            "dropout": 0.12,
            "use_query_gate": False,
            "query_gate_type": "channel",
        },
    },
    "tkgl-polecat": {
        "time_mrr": {
            "topk": 80,
            "multi_windows": "30",
            "train_num_neg": 2,
            "selection_metric": "mrr",
            "event_dim": 64,
            "hidden_dim": 128,
            "dropout": 0.15,
            "use_query_gate": False,
            "query_gate_type": "channel",
        },
    },
    "tkgl-icews": {
        "time_mrr": {
            "topk": 60,
            "multi_windows": "30",
            "train_num_neg": 2,
            "selection_metric": "mrr",
            "event_dim": 64,
            "hidden_dim": 128,
            "dropout": 0.15,
            "use_query_gate": False,
            "query_gate_type": "channel",
        },
    },
}

YELP_TIME_PRESET = {
    "time_yelp_geo": {
        "topk": 40,
        "multi_windows": "10,40",
        "train_num_neg": 8,
        "selection_metric": "hit10",
        "event_dim": 96,
        "hidden_dim": 160,
        "dropout": 0.15,
        "use_query_gate": True,
        "query_gate_type": "channel",
        "use_node_geo": True,
        "thg_time_days": True,
        "user_center_half_life_days": 365.0,
        "abs_time_periods": "7,30,180,365",
    },
}
for _dataset in YELP_DATASETS:
    TIME_PRESETS[_dataset] = copy.deepcopy(YELP_TIME_PRESET)


def load_hybrid_time_dirs(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "new_hybrid_inputs_v1":
        raise ValueError(f"unsupported hybrid config format: {payload.get('format')!r}")
    return {
        dataset: {entry["id"]: entry["dir"] for entry in entries}
        for dataset, entries in payload.get("time_runs", {}).items()
    }


def parse_value(text):
    lowered = str(text).strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_overrides(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override has empty key: {item!r}")
        overrides[key] = parse_value(value)
    return overrides


def build_time_args(cli, preset_id, preset, output_dir):
    defaults = DATASET_DEFAULTS[cli.dataset]
    payload = {
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "gpu": int(cli.gpu),
        "batch_size": int(defaults["batch_size"]),
        "eval_batch_size": int(defaults["eval_batch_size"]),
        "ns_q": int(defaults["ns_q"]),
        "ns_seed": int(cli.ns_seed),
        "train_predict_ratio": float(cli.train_predict_ratio),
        "cache_eval_source": bool(cli.cache_eval_source),
        "topk": 15,
        "train_num_neg": 4,
        "hard_neg_ratio": float(cli.hard_neg_ratio),
        "train_sampler": "grouped_exact",
        "train_group_matrix_mb": float(defaults["train_group_matrix_mb"]),
        "use_neighbor_id": True,
        "use_abs_time": True,
        "abs_time_periods": "1,7,30,365",
        "abs_time_harmonics": 1,
        "abs_time_use_raw": False,
        "use_node_geo": False,
        "thg_time_days": False,
        "user_center_half_life_days": 365.0,
        "use_query_gate": False,
        "query_gate_type": "channel",
        "use_rank_pos": True,
        "multi_windows": "",
        "use_cross_history": False,
        "cross_heads": 2,
        "event_encoder": "mixer",
        "transformer_heads": 2,
        "transformer_ff_dim": None,
        "time_dim": 64,
        "rel_dim": 64,
        "node_dim": 64,
        "event_dim": 64,
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.1,
        "time_min": 1.0,
        "token_expansion_factor": 0.5,
        "channel_expansion_factor": 4.0,
        "use_single_layer": False,
        "predictor_mode": "diag",
        "num_epochs": int(defaults["num_epochs"]),
        "patience": int(defaults["patience"]),
        "selection_metric": "mrr",
        "quick_val_events": int(cli.quick_val_events),
        "quick_val_fraction": 0.0 if cli.full_val_selection else float(defaults["quick_val_fraction"]),
        "lr": float(cli.lr),
        "weight_decay": float(cli.weight_decay),
        "temperature": 1.0,
        "train_loss": "margin",
        "rank_margin": 1.0,
        "grad_clip": 1.0,
        "tolerance": 1e-8,
        "curriculum_decay": 0.0,
        "curriculum_raw_age": False,
        "eval_neg_chunk": int(defaults["eval_neg_chunk"]),
        "max_eval_pairs": int(defaults["max_eval_pairs"]),
        "eval_node_preload_chunk": int(cli.eval_node_preload_chunk),
        "preload_eval_nodes": not bool(cli.no_preload_eval_nodes),
        "dense_eval_node_cache": not bool(cli.no_dense_eval_node_cache),
        "max_eval_node_cache_mb": float(cli.max_eval_node_cache_mb),
        "profile_sync": bool(cli.profile_sync),
        "eval_test": not bool(cli.no_eval_test),
        "force": bool(cli.force),
        "no_retrain_on_train_prefix": bool(cli.no_retrain_on_train_prefix),
        "output_dir": (
            output_dir
            if cli.use_hybrid_config_dir and not bool(cli.no_retrain_on_train_prefix)
            else ""
        ),
        "time_preset_id": preset_id,
    }
    payload.update(preset)
    if cli.user_center_half_life_days is not None:
        payload["user_center_half_life_days"] = float(cli.user_center_half_life_days)
    payload.update(parse_overrides(cli.override))
    if cli.full_val_selection and cli.use_hybrid_config_dir:
        raise ValueError(
            "--full_val_selection changes the run protocol; omit --use_hybrid_config_dir "
            "or update configs/new_hybrid_inputs.json to the new output directory."
        )
    return SimpleNamespace(**payload)


def ensure_score_files(out_dir, modes):
    missing = []
    for mode in modes:
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
            path = osp.join(out_dir, f"{mode}_{suffix}")
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(f"time score generation incomplete; first missing file: {missing[0]}")


def run_one(cli, preset_id, hybrid_dirs):
    presets = TIME_PRESETS.get(cli.dataset, {})
    if preset_id not in presets:
        raise ValueError(f"unknown preset {preset_id!r} for dataset {cli.dataset!r}; choices={sorted(presets)}")
    output_dir = hybrid_dirs.get(cli.dataset, {}).get(preset_id, "")
    if cli.use_hybrid_config_dir and not output_dir:
        print(
            f"[TrainTime] preset={preset_id} is not present in {cli.hybrid_config}; "
            "using the natural new_single_pipeline.time output directory",
            flush=True,
        )
    args = build_time_args(cli, preset_id, copy.deepcopy(presets[preset_id]), output_dir)
    out_dir = time.get_out_dir(args)
    print(f"[TrainTime] dataset={cli.dataset} preset={preset_id} -> {out_dir}", flush=True)
    result = time.main(args)
    modes = ["val"]
    if float(args.train_predict_ratio) > 0.0:
        modes.insert(0, "train")
    if getattr(args, "eval_test", True):
        modes.append("test")
    ensure_score_files(out_dir, modes)
    print(f"[TrainTime] complete preset={preset_id} result={result} out_dir={out_dir}", flush=True)
    return out_dir


def resolve_preset_id(cli):
    presets = TIME_PRESETS.get(cli.dataset, {})
    if not presets:
        raise ValueError(f"no time presets are registered for dataset {cli.dataset!r}")
    if cli.preset:
        if cli.preset not in presets:
            raise ValueError(
                f"unknown preset {cli.preset!r} for dataset {cli.dataset!r}; "
                f"choices={sorted(presets)}"
            )
        return cli.preset
    return next(iter(presets))


def parse_args():
    parser = argparse.ArgumentParser(
        "Run new_single_pipeline.time and write train/val/test score stores for new hybrid."
    )
    parser.add_argument("--dataset", choices=sorted(TIME_PRESETS), default="ICEWS14")
    parser.add_argument(
        "--preset",
        default="",
        help="Preset id from train_time.py TIME_PRESETS. Empty means the dataset default.",
    )
    parser.add_argument("--run_all_for_dataset", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--hybrid_config", default=HYBRID_CONFIG)
    parser.add_argument("--use_hybrid_config_dir", action="store_true", default=True)
    parser.add_argument("--no_use_hybrid_config_dir", dest="use_hybrid_config_dir", action="store_false")
    parser.add_argument("--no_retrain_on_train_prefix", action="store_true", default=False)
    parser.add_argument("--full_val_selection", action="store_true", default=False)
    parser.add_argument("--quick_val_events", type=int, default=0)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument(
        "--user_center_half_life_days",
        type=float,
        default=None,
        help="Override the Yelp user activity-center exponential half-life in days.",
    )
    parser.add_argument("--eval_node_preload_chunk", type=int, default=65536)
    parser.add_argument("--max_eval_node_cache_mb", type=float, default=4096.0)
    parser.add_argument("--cache_eval_source", action="store_true", default=False)
    parser.add_argument("--no_preload_eval_nodes", action="store_true", default=False)
    parser.add_argument("--no_dense_eval_node_cache", action="store_true", default=False)
    parser.add_argument("--profile_sync", action="store_true", default=False)
    parser.add_argument("--no_eval_test", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override any new_single_pipeline.time arg as key=value; may be repeated.",
    )
    return parser.parse_args()


def main(cli):
    hybrid_dirs = load_hybrid_time_dirs(cli.hybrid_config)
    if cli.run_all_for_dataset:
        preset_ids = list(TIME_PRESETS[cli.dataset])
    else:
        preset_ids = [resolve_preset_id(cli)]
    return [run_one(cli, preset_id, hybrid_dirs) for preset_id in preset_ids]


if __name__ == "__main__":
    main(parse_args())
