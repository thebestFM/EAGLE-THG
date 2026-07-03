import argparse
import gc
import hashlib
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np

import train_time
from new_single_pipeline.structure_lgbm import (
    BConfig,
    RescueHybridFeatureBuilder,
    build_rescue_hybrid_matrix,
    ensure_dir,
    evaluate_rescue_hybrid_model,
    evaluate_score_store,
    fit_lgbm_ranker,
    format_metrics,
    metric_value,
    save_component_score_stores,
    save_lgbm_model,
)
import train_new_structure as tns
from new_single_pipeline import time as time_module
from utils import describe_loaded_data, load_datasets, ranking_metric_key, save_config, save_metrics, set_random_seed


PROTOCOL = "new_hybrid_save_top10_rescue_topk_v1"
SUPPORTED_STRUCTURE_IMPLS = {"new_structure_v3", "new_structure_v4"}
BEST_HYPER_DIR = "best_hyper"


HYBRID_PARAM_PRESETS = [
    {
        "n_estimators": 1,
        "learning_rate": 0.03753885387101247,
        "num_leaves": 21,
        "max_depth": 12,
        "min_child_samples": 39,
        "reg_lambda": 0.6376811127061687,
        "reg_alpha": 0.0013242769886098894,
        "min_split_gain": 0.00023728487317629075,
        "subsample": 0.990325975281362,
        "colsample_bytree": 0.7627169011118449,
    },
    {
        "n_estimators": 2,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "max_depth": 8,
        "min_child_samples": 60,
        "reg_lambda": 0.8,
        "reg_alpha": 0.01,
        "min_split_gain": 0.001,
        "subsample": 0.95,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 4,
        "learning_rate": 0.015,
        "num_leaves": 31,
        "max_depth": 10,
        "min_child_samples": 100,
        "reg_lambda": 1.5,
        "reg_alpha": 0.05,
        "min_split_gain": 0.005,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 8,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 8,
        "min_child_samples": 80,
        "reg_lambda": 1.0,
        "reg_alpha": 0.01,
        "min_split_gain": 0.001,
        "subsample": 0.95,
        "colsample_bytree": 0.85,
    },
    {
        "n_estimators": 16,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 10,
        "min_child_samples": 120,
        "reg_lambda": 1.5,
        "reg_alpha": 0.03,
        "min_split_gain": 0.003,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 32,
        "learning_rate": 0.015,
        "num_leaves": 47,
        "max_depth": 10,
        "min_child_samples": 150,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "min_split_gain": 0.005,
        "subsample": 0.9,
        "colsample_bytree": 0.75,
    },
    {
        "n_estimators": 64,
        "learning_rate": 0.01,
        "num_leaves": 63,
        "max_depth": 12,
        "min_child_samples": 200,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "min_split_gain": 0.01,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
    },
    {
        "n_estimators": 128,
        "learning_rate": 0.006,
        "num_leaves": 63,
        "max_depth": 12,
        "min_child_samples": 300,
        "reg_lambda": 5.0,
        "reg_alpha": 0.2,
        "min_split_gain": 0.02,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
    },
    {
        "n_estimators": 24,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 7,
        "min_child_samples": 250,
        "reg_lambda": 4.0,
        "reg_alpha": 0.1,
        "min_split_gain": 0.02,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
]


def stable_hash(payload, length=12):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[: int(length)]


def resolve_time_dir(raw_dir, time_root):
    if not time_root:
        return raw_dir
    rel = raw_dir.replace("\\", "/")
    parts = rel.split("/")
    if parts and parts[0] == "results_time_tkg_single":
        rel = "/".join(parts[1:])
    return osp.join(time_root, rel)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_best_hyper_path(dataset):
    return osp.join(BEST_HYPER_DIR, f"{dataset}.json")


def apply_mapping(args, payload, prefix=""):
    for key, value in payload.items():
        if isinstance(value, dict):
            continue
        name = f"{prefix}{key}" if prefix else key
        if not hasattr(args, name):
            raise ValueError(f"best_hyper contains unknown option {name!r}")
        setattr(args, name, value)


def apply_best_hyper(args):
    path = str(getattr(args, "best_hyper", "") or "").strip()
    if not path:
        return None
    if path.lower() == "auto":
        path = default_best_hyper_path(args.dataset)
    if not osp.isfile(path):
        raise FileNotFoundError(f"best_hyper file not found: {path}")
    payload = load_json(path)
    if payload.get("format") != "best_hyper_v1":
        raise ValueError(f"unsupported best_hyper format in {path}: {payload.get('format')!r}")
    if payload.get("dataset") and str(payload["dataset"]) != str(args.dataset):
        args.dataset = str(payload["dataset"])
    apply_mapping(args, payload.get("common", {}))
    apply_mapping(args, payload.get("time", {}), "time_")
    apply_mapping(args, payload.get("structure", {}), "structure_")
    apply_mapping(args, payload.get("hybrid", {}))
    args.best_hyper = path
    return payload


def build_time_args(args):
    values = {
        "dataset": args.dataset,
        "seed": args.seed,
        "gpu": 0,
        "output_dir": args.time_dir,
        "force": False,
        "batch_size": args.time_batch_size,
        "eval_batch_size": args.time_eval_batch_size,
        "eval_neg_chunk": args.time_eval_neg_chunk,
        "max_eval_pairs": args.time_max_eval_pairs,
        "stream_eval_batch_events": args.time_stream_eval_batch_events,
        "eval_node_preload_chunk": args.time_eval_node_preload_chunk,
        "max_eval_node_cache_mb": args.time_max_eval_node_cache_mb,
        "preload_eval_nodes": args.time_preload_eval_nodes,
        "dense_eval_node_cache": args.time_dense_eval_node_cache,
        "cache_eval_source": args.time_cache_eval_source,
        "eval_test": True,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "quick_val_events": args.time_quick_val_events,
        "quick_val_fraction": args.time_quick_val_fraction,
        "num_epochs": args.time_num_epochs,
        "patience": args.time_patience,
        "selection_metric": args.time_selection_metric,
        "lr": args.time_lr,
        "weight_decay": args.time_weight_decay,
        "train_num_neg": args.time_train_num_neg,
        "stream_train_batch_events": args.time_stream_train_batch_events,
        "hard_neg_ratio": args.time_hard_neg_ratio,
        "train_sampler": args.time_train_sampler,
        "train_group_matrix_mb": args.time_train_group_matrix_mb,
        "train_loss": args.time_train_loss,
        "rank_margin": args.time_rank_margin,
        "temperature": args.time_temperature,
        "grad_clip": args.time_grad_clip,
        "tolerance": args.time_tolerance,
        "curriculum_decay": args.time_curriculum_decay,
        "curriculum_raw_age": args.time_curriculum_raw_age,
        "topk": args.time_topk,
        "multi_windows": args.time_multi_windows,
        "time_dim": args.time_time_dim,
        "rel_dim": args.time_rel_dim,
        "node_dim": args.time_node_dim,
        "event_dim": args.time_event_dim,
        "hidden_dim": args.time_hidden_dim,
        "num_layers": args.time_num_layers,
        "dropout": args.time_dropout,
        "time_min": args.time_time_min,
        "token_expansion_factor": args.time_token_expansion_factor,
        "channel_expansion_factor": args.time_channel_expansion_factor,
        "use_single_layer": args.time_use_single_layer,
        "predictor_mode": args.time_predictor_mode,
        "event_encoder": args.time_event_encoder,
        "transformer_heads": args.time_transformer_heads,
        "transformer_ff_dim": args.time_transformer_ff_dim,
        "use_cross_history": args.time_use_cross_history,
        "cross_heads": args.time_cross_heads,
        "use_neighbor_id": args.time_use_neighbor_id,
        "use_abs_time": args.time_use_abs_time,
        "abs_time_periods": args.time_abs_time_periods,
        "abs_time_harmonics": args.time_abs_time_harmonics,
        "abs_time_use_raw": args.time_abs_time_use_raw,
        "use_query_gate": args.time_use_query_gate,
        "query_gate_type": args.time_query_gate_type,
        "use_rank_pos": args.time_use_rank_pos,
        "use_node_geo": args.time_use_node_geo,
        "thg_time_days": args.time_thg_time_days,
        "user_center_half_life_days": args.time_user_center_half_life_days,
        "no_retrain_on_train_prefix": args.time_no_retrain_on_train_prefix,
        "reuse_no_retrain_full": True,
        "profile_sync": False,
        "use_amp": args.time_use_amp,
        "allow_tf32": args.time_allow_tf32,
    }
    return train_time.normalize_args(SimpleNamespace(**values))


def locate_time_dir(args):
    if args.time_dir:
        return resolve_time_dir(args.time_dir, args.time_root)
    return resolve_time_dir(time_module.get_out_dir(build_time_args(args)), args.time_root)


def structure_config_from_args(args):
    return {
        "id": args.structure_id or "structure_args",
        "batch_size": args.structure_batch_size,
        "max_events_in_single_batch": args.structure_max_events_in_single_batch,
        "dict_mode": args.structure_dict_mode,
        "shared_w": args.structure_shared_w,
        "per_rel_use_mtrans": args.structure_per_rel_use_mtrans,
        "ppr_k": args.structure_ppr_k,
        "top_k_relation": args.structure_top_k_relation,
        "ppr_alpha": args.structure_ppr_alpha,
        "ppr_beta": args.structure_ppr_beta,
        "gamma": args.structure_gamma,
        "direct_single_hop": args.structure_direct_single_hop,
        "decay_direct": args.structure_decay_direct,
        "top_share": args.structure_top_share,
        "top_direct": args.structure_top_direct,
        "decay_rel_trans": args.structure_decay_rel_trans,
        "window_semantic_sim": args.structure_window_semantic_sim,
        "window_trans": args.structure_window_trans,
        "close_update_backward": args.structure_close_update_backward,
        "output_root": args.structure_output_root,
    }


def make_structure_args(args, cfg):
    payload = dict(cfg)
    payload.update(
        {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "ns_q": int(args.ns_q),
            "ns_seed": int(args.ns_seed),
            "train_predict_ratio": float(args.train_predict_ratio),
            "batch_size": int(cfg["batch_size"]),
            "query_batch_size": int(args.query_batch_size),
            "source_join_threads": int(args.source_join_threads),
            "source_join_log_batches": int(args.source_join_log_batches),
            "dsh_log_bucket_stats": bool(args.dsh_log_bucket_stats),
            "b_cfg": BConfig(
                mode=args.b_mode,
                binary_unseen=float(args.b_binary_unseen),
                continuous_alpha=float(args.b_continuous_alpha),
            ),
        }
    )
    sargs = SimpleNamespace(**payload)
    if hasattr(tns, "normalize_args"):
        tns.normalize_args(sargs)
    return sargs


def locate_structure_dir(args, sargs):
    if args.structure_dir:
        return args.structure_dir
    return tns.make_new_result_dir(sargs)


def make_out_dir(args, run_spec):
    h = stable_hash(
        {
            "protocol": PROTOCOL,
            "dataset": args.dataset,
            "run_spec": run_spec,
            "structure_impl": getattr(tns, "NEW_STRUCTURE_IMPL", ""),
            "seed": args.seed,
            "ns_seed": args.ns_seed,
            "train_predict_ratio": args.train_predict_ratio,
            "rescue_topk": args.rescue_topk,
            "rescue_min_pos_rank": args.rescue_min_pos_rank,
            "rescue_max_pos_rank": args.rescue_max_pos_rank,
            "rescue_exclude_top10": args.rescue_exclude_top10,
            "hybrid_select_split": args.hybrid_select_split,
            "hybrid_preset_indices": args.hybrid_preset_indices,
            "component_splits": args.component_splits,
            "skip_time_score_check": args.skip_time_score_check,
            "skip_component_metrics": args.skip_component_metrics,
            "skip_save_top10": args.skip_save_top10,
            "max_hybrid_train_queries": args.max_hybrid_train_queries,
            "hybrid_train_query_stride": args.hybrid_train_query_stride,
            "hybrid_preset_hash": stable_hash(HYBRID_PARAM_PRESETS, length=10),
            "focus_metric": args.focus_metric,
            "b": {
                "mode": args.b_mode,
                "binary_unseen": args.b_binary_unseen,
                "continuous_alpha": args.b_continuous_alpha,
            },
        },
        12,
    )
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"save_top10_{h}")


def require_time_scores(time_dir, label, splits=("train", "val", "test")):
    missing = []
    for split in splits:
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
            path = osp.join(time_dir, f"{split}_{suffix}")
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(f"{label} missing time score files, first missing: {missing[0]}")


def require_structure_scores(structure_dir, label, splits=("train", "val", "test")):
    missing = []
    for split in splits:
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
            path = osp.join(structure_dir, f"{split}_{suffix}")
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(f"{label} missing structure score files, first missing: {missing[0]}")


def parse_component_splits(args):
    raw = str(getattr(args, "component_splits", "") or "").strip()
    if raw:
        splits = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        splits = ["train", str(args.hybrid_select_split), "test"]
    ordered = []
    for split in splits:
        if split not in ("train", "val", "test"):
            raise ValueError("--component_splits entries must be train,val,test")
        if split not in ordered:
            ordered.append(split)
    for split in ("train", str(args.hybrid_select_split), "test"):
        if split not in ordered:
            ordered.append(split)
    return ordered


def selected_hybrid_presets(args):
    raw = str(getattr(args, "hybrid_preset_indices", "") or "").strip()
    if not raw:
        return [(idx, params) for idx, params in enumerate(HYBRID_PARAM_PRESETS, start=1)]
    indices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx <= 0 or idx > len(HYBRID_PARAM_PRESETS):
            raise ValueError(f"--hybrid_preset_indices contains invalid preset index {idx}")
        if idx not in indices:
            indices.append(idx)
    if not indices:
        raise ValueError("--hybrid_preset_indices did not contain any preset index")
    return [(idx, HYBRID_PARAM_PRESETS[idx - 1]) for idx in indices]


def train_best_rescue_hybrid(data, sargs, args, device, out_dir, struct_id, time_run, component_dir):
    num_rels = tns.runtime_num_rels(data) if hasattr(tns, "runtime_num_rels") else data["num_rels"]
    rescue_feature_builder = RescueHybridFeatureBuilder(num_rels)
    time_dir = time_run["dir"]
    include_top10 = not bool(getattr(args, "rescue_exclude_top10", False))
    print(
        f"[SaveTop10][rescue] build train matrix struct={struct_id} time={time_run['id']} "
        f"topk={int(args.rescue_topk)} pos_rank={int(args.rescue_min_pos_rank)}..{int(args.rescue_max_pos_rank)} "
        f"include_top10={include_top10}",
        flush=True,
    )
    X_train, y_train, group, train_info = build_rescue_hybrid_matrix(
        data,
        "train",
        sargs,
        rescue_feature_builder,
        time_dir,
        device,
        int(args.rescue_topk),
        component_root=component_dir,
        min_pos_rank=int(args.rescue_min_pos_rank),
        max_pos_rank=int(args.rescue_max_pos_rank),
        include_top10=include_top10,
        max_queries=int(args.max_hybrid_train_queries),
        query_stride=int(args.hybrid_train_query_stride),
    )
    print(
        f"[SaveTop10][rescue] train rows={train_info['rows']} queries={train_info['queries']} "
        f"preserve={train_info['preserve_queries']} rescue={train_info['rescue_queries']} "
        f"skipped_pos_after_topk={train_info['skipped_pos_after_topk']} features={X_train.shape[1]}",
        flush=True,
    )
    best = None
    records = []
    select_split = str(args.hybrid_select_split)
    for idx, params in selected_hybrid_presets(args):
        print(f"[SaveTop10][rescue] preset {idx} params={params}", flush=True)
        model = fit_lgbm_ranker(
            X_train,
            y_train,
            group,
            rescue_feature_builder.feature_names,
            [],
            args,
            params,
        )
        select_metrics = evaluate_rescue_hybrid_model(
            data,
            select_split,
            sargs,
            rescue_feature_builder,
            model,
            time_dir,
            device,
            int(args.rescue_topk),
            component_root=component_dir,
        )
        score = metric_value(select_metrics, args.focus_metric)
        print(
            f"[SaveTop10][rescue] preset {idx} {select_split} {format_metrics(select_metrics)} "
            f"score={score:.5f} stats={select_metrics.get('rescue_stats')}",
            flush=True,
        )
        rec = {
            "preset": idx,
            "params": dict(params),
            "selection_split": select_split,
            "selection_metrics": select_metrics,
            "score": float(score),
            "train_info": train_info,
        }
        records.append(rec)
        if best is None or score > best["score"]:
            if best is not None:
                del best["model"]
                gc.collect()
            best = {"model": model, "score": float(score), "record": rec}
        else:
            del model
            gc.collect()
    del X_train, y_train, group
    gc.collect()
    pair_id = f"{struct_id}__{time_run['id']}__rescue_top{int(args.rescue_topk)}"
    model_path = osp.join(out_dir, "rescue_models", f"{pair_id}.txt")
    save_lgbm_model(best["model"], model_path)
    top10_path = None
    if str(select_split) == "test" and bool(args.skip_save_top10):
        test_metrics = best["record"]["selection_metrics"]
    else:
        if not bool(args.skip_save_top10):
            top10_path = osp.join(out_dir, "top10", f"{pair_id}.test_top10.jsonl")
        test_metrics = evaluate_rescue_hybrid_model(
            data,
            "test",
            sargs,
            rescue_feature_builder,
            best["model"],
            time_dir,
            device,
            int(args.rescue_topk),
            component_root=component_dir,
            save_top10_path=top10_path,
        )
    print(
        f"[SaveTop10][rescue] best pair={pair_id} test {format_metrics(test_metrics)} "
        f"stats={test_metrics.get('rescue_stats')} top10={top10_path}",
        flush=True,
    )
    best["record"].update(
        {
            "test_metrics": test_metrics,
            "model_path": model_path,
            "top10_path": top10_path,
            "pair_id": pair_id,
            "struct_id": struct_id,
            "time_id": time_run["id"],
            "time_dir": time_dir,
            "rescue_topk": int(args.rescue_topk),
        }
    )
    return best["record"], records


def validate_args(args):
    impl = getattr(tns, "NEW_STRUCTURE_IMPL", None)
    if impl not in SUPPORTED_STRUCTURE_IMPLS:
        raise RuntimeError(
            "train_new_hybrid_save_top10.py requires train_new_structure.py "
            f"impl in {sorted(SUPPORTED_STRUCTURE_IMPLS)}; got {impl!r}"
        )
    focus = str(args.focus_metric).upper().replace("@", "")
    aliases = {"MRR": "mrr", "H1": "hr1", "HR1": "hr1", "H10": "hr10", "HR10": "hr10"}
    if focus not in aliases:
        raise ValueError("--focus_metric must be one of MRR/H1/H10")
    args.focus_metric = aliases[focus]
    ranking_metric_key(args.focus_metric, strict=True)
    if int(args.query_batch_size) <= 0:
        raise ValueError("--query_batch_size must be > 0")
    if int(args.rescue_topk) <= 0:
        raise ValueError("--rescue_topk must be > 0")
    if int(args.rescue_min_pos_rank) <= 0 or int(args.rescue_max_pos_rank) < int(args.rescue_min_pos_rank):
        raise ValueError("--rescue_min_pos_rank/--rescue_max_pos_rank must define a positive rank interval")
    if int(args.rescue_max_pos_rank) > int(args.rescue_topk):
        raise ValueError("--rescue_max_pos_rank cannot exceed --rescue_topk")
    if str(args.hybrid_select_split) not in ("val", "test"):
        raise ValueError("--hybrid_select_split must be val or test")
    if int(args.max_hybrid_train_queries) < 0:
        raise ValueError("--max_hybrid_train_queries must be >= 0")
    if int(args.hybrid_train_query_stride) <= 0:
        raise ValueError("--hybrid_train_query_stride must be > 0")
    selected_hybrid_presets(args)
    parse_component_splits(args)
    if int(args.num_threads) <= 0:
        raise ValueError("--num_threads must be > 0")
    if int(args.source_join_threads) < 0:
        raise ValueError("--source_join_threads must be >= 0")
    if str(args.b_mode) == "continuous" and float(args.b_continuous_alpha) < 0.0:
        raise ValueError("--b_continuous_alpha must be >= 0")
    if int(args.ns_q) <= 0 and int(args.ns_q) != -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if int(args.structure_batch_size) <= 0:
        raise ValueError("--structure_batch_size must be > 0")
    if int(args.structure_max_events_in_single_batch) <= 0:
        raise ValueError("--structure_max_events_in_single_batch must be > 0")


def run(args):
    best_payload = apply_best_hyper(args)
    validate_args(args)
    set_random_seed(args.seed)
    device = "cpu"
    cfg = structure_config_from_args(args)
    sargs = make_structure_args(args, cfg)
    time_dir = locate_time_dir(args)
    structure_dir = locate_structure_dir(args, sargs)
    component_splits = parse_component_splits(args)
    require_time_scores(time_dir, f"time {args.time_id or time_dir}", splits=component_splits)
    require_structure_scores(structure_dir, f"structure {cfg['id']}", splits=("train", "val", "test"))
    time_run = {
        "id": args.time_id or osp.basename(osp.normpath(time_dir)),
        "dir": time_dir,
    }
    run_spec = {
        "best_hyper": args.best_hyper,
        "best_hyper_hash": stable_hash(best_payload, length=10) if best_payload is not None else "",
        "time_dir": time_dir,
        "structure_dir": structure_dir,
        "structure_config": cfg,
    }
    out_dir = ensure_dir(make_out_dir(args, run_spec))
    print(f"[SaveTop10] output -> {out_dir}", flush=True)
    print(
        f"[SaveTop10] protocol={PROTOCOL} dataset={args.dataset} ns_q={args.ns_q} "
        f"focus={args.focus_metric} device={device}",
        flush=True,
    )
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[SaveTop10]")

    if bool(args.skip_time_score_check):
        print(f"[SaveTop10][time] {time_run['id']} score-check skipped dir={time_run['dir']}", flush=True)
    else:
        test_metrics = evaluate_score_store(time_run["dir"], data, "test", SimpleNamespace(query_batch_size=args.query_batch_size))
        time_run["computed_test_metrics"] = test_metrics
        print(f"[SaveTop10][time] {time_run['id']} test {format_metrics(test_metrics)} dir={time_run['dir']}", flush=True)

    structure_records = []
    pair_records = []
    best_pair = None
    struct_id = cfg["id"]
    print(f"[SaveTop10] structure config {struct_id}: {cfg}", flush=True)
    print(f"[SaveTop10] structure score dir verified: {structure_dir}", flush=True)
    t0 = time.time()
    component_dir, component_metrics = save_component_score_stores(
        data,
        sargs,
        device,
        out_dir,
        struct_id,
        splits=component_splits,
        compute_metrics=not bool(args.skip_component_metrics),
    )
    raw_test = None
    if not bool(args.skip_component_metrics):
        raw_test = component_metrics["splits"]["test"]["metrics"]["structure_raw"]
        print(
            f"[SaveTop10][structure_raw] struct={struct_id} test {format_metrics(raw_test)} "
            f"component_scores={component_dir}",
            flush=True,
        )
    else:
        print(f"[SaveTop10][structure_raw] struct={struct_id} metrics=skipped component_scores={component_dir}", flush=True)
    struct_record = {
        "id": struct_id,
        "config": cfg,
        "structure_dir": structure_dir,
        "mode": "structure_components_from_args",
        "component_score_dir": component_dir,
        "component_metrics": component_metrics,
        "elapsed_s": time.time() - t0,
    }
    if raw_test is not None:
        struct_record["train_new_structure_style_test_metrics"] = raw_test
    structure_records.append(struct_record)
    pair_t0 = time.time()
    best_record, all_records = train_best_rescue_hybrid(
        data,
        sargs,
        args,
        device,
        out_dir,
        struct_id,
        time_run,
        component_dir,
    )
    best_record["all_rescue_presets"] = all_records
    best_record["elapsed_s"] = time.time() - pair_t0
    pair_records.append(best_record)
    best_pair = best_record
    gc.collect()

    summary = {
        "format": "new_hybrid_save_top10_summary_v1",
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "args": vars(args).copy(),
        "run_spec": run_spec,
        "time_runs": [time_run],
        "structure_records": structure_records,
        "pair_records": sorted(
            pair_records,
            key=lambda r: metric_value(r["test_metrics"], args.focus_metric),
            reverse=True,
        ),
        "best": best_pair,
    }
    save_config(out_dir, summary["args"])
    save_metrics(out_dir, summary)
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[SaveTop10] best pair={best_pair['pair_id']} "
        f"test {format_metrics(best_pair['test_metrics'])} top10={best_pair['top10_path']}",
        flush=True,
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser("Rescue-style topK hybrid reranker that saves test top10 per query.")
    parser.add_argument("--dataset", default="ICEWS14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--best_hyper", default="", help="Path to best_hyper JSON, or 'auto' for best_hyper/<dataset>.json.")
    parser.add_argument("--time_dir", default="", help="Explicit time score directory. If empty, it is derived from time_* args.")
    parser.add_argument("--time_id", default="")
    parser.add_argument("--time_root", default="")
    parser.add_argument("--structure_dir", default="", help="Explicit structure score directory. If empty, it is derived from structure_* args.")
    parser.add_argument("--structure_id", default="")
    parser.add_argument("--structure_output_root", default="results_new_structure")
    parser.add_argument("--output_root", default="results_new_hybrid_save_top10")

    parser.add_argument("--time_batch_size", type=int, default=1024)
    parser.add_argument("--time_eval_batch_size", type=int, default=256)
    parser.add_argument("--time_eval_neg_chunk", type=int, default=128)
    parser.add_argument("--time_max_eval_pairs", type=int, default=125000)
    parser.add_argument("--time_stream_eval_batch_events", type=int, default=32)
    parser.add_argument("--time_eval_node_preload_chunk", type=int, default=65536)
    parser.add_argument("--time_max_eval_node_cache_mb", type=float, default=4096.0)
    parser.set_defaults(time_preload_eval_nodes=True, time_dense_eval_node_cache=True)
    parser.add_argument("--time_no_preload_eval_nodes", dest="time_preload_eval_nodes", action="store_false")
    parser.add_argument("--time_no_dense_eval_node_cache", dest="time_dense_eval_node_cache", action="store_false")
    parser.add_argument("--time_cache_eval_source", action="store_true", default=False)
    parser.add_argument("--time_quick_val_events", type=int, default=0)
    parser.add_argument("--time_quick_val_fraction", type=float, default=0.2)
    parser.add_argument("--time_num_epochs", type=int, default=20)
    parser.add_argument("--time_patience", type=int, default=4)
    parser.add_argument("--time_selection_metric", choices=("mrr", "hit10"), default="mrr")
    parser.add_argument("--time_lr", type=float, default=8e-4)
    parser.add_argument("--time_weight_decay", type=float, default=5e-5)
    parser.add_argument("--time_train_num_neg", type=int, default=8)
    parser.add_argument("--time_stream_train_batch_events", type=int, default=2048)
    parser.add_argument("--time_hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--time_train_sampler", choices=("exact", "grouped_exact", "fast"), default="grouped_exact")
    parser.add_argument("--time_train_group_matrix_mb", type=float, default=512.0)
    parser.add_argument("--time_train_loss", choices=("margin", "ce"), default="margin")
    parser.add_argument("--time_rank_margin", type=float, default=1.0)
    parser.add_argument("--time_temperature", type=float, default=1.0)
    parser.add_argument("--time_grad_clip", type=float, default=1.0)
    parser.add_argument("--time_tolerance", type=float, default=1e-8)
    parser.add_argument("--time_curriculum_decay", type=float, default=0.0)
    parser.add_argument("--time_curriculum_raw_age", action="store_true", default=False)
    parser.add_argument("--time_topk", type=int, default=40)
    parser.add_argument("--time_multi_windows", default="10,40")
    parser.add_argument("--time_time_dim", type=int, default=64)
    parser.add_argument("--time_rel_dim", type=int, default=64)
    parser.add_argument("--time_node_dim", type=int, default=64)
    parser.add_argument("--time_event_dim", type=int, default=96)
    parser.add_argument("--time_hidden_dim", type=int, default=160)
    parser.add_argument("--time_num_layers", type=int, default=1)
    parser.add_argument("--time_dropout", type=float, default=0.15)
    parser.add_argument("--time_time_min", type=float, default=1.0)
    parser.add_argument("--time_token_expansion_factor", type=float, default=0.5)
    parser.add_argument("--time_channel_expansion_factor", type=float, default=4.0)
    parser.add_argument("--time_use_single_layer", action="store_true", default=False)
    parser.add_argument("--time_predictor_mode", choices=("diag", "concat"), default="diag")
    parser.add_argument("--time_event_encoder", choices=("mixer", "transformer"), default="mixer")
    parser.add_argument("--time_transformer_heads", type=int, default=2)
    parser.add_argument("--time_transformer_ff_dim", type=int, default=None)
    parser.add_argument("--time_use_cross_history", action="store_true", default=False)
    parser.add_argument("--time_cross_heads", type=int, default=2)
    parser.set_defaults(time_use_neighbor_id=True, time_use_abs_time=True, time_use_query_gate=True, time_use_rank_pos=True)
    parser.add_argument("--time_no_use_neighbor_id", dest="time_use_neighbor_id", action="store_false")
    parser.add_argument("--time_no_use_abs_time", dest="time_use_abs_time", action="store_false")
    parser.add_argument("--time_abs_time_periods", default="7,30,180,365")
    parser.add_argument("--time_abs_time_harmonics", type=int, default=1)
    parser.add_argument("--time_abs_time_use_raw", action="store_true", default=False)
    parser.add_argument("--time_no_use_query_gate", dest="time_use_query_gate", action="store_false")
    parser.add_argument("--time_query_gate_type", choices=("channel", "scalar"), default="channel")
    parser.add_argument("--time_no_use_rank_pos", dest="time_use_rank_pos", action="store_false")
    parser.add_argument("--time_use_node_geo", dest="time_use_node_geo", action="store_true", default=None)
    parser.add_argument("--time_no_use_node_geo", dest="time_use_node_geo", action="store_false")
    parser.add_argument("--time_thg_time_days", dest="time_thg_time_days", action="store_true", default=None)
    parser.add_argument("--time_no_thg_time_days", dest="time_thg_time_days", action="store_false")
    parser.add_argument("--time_user_center_half_life_days", type=float, default=365.0)
    parser.add_argument("--time_no_retrain_on_train_prefix", action="store_true", default=False)
    parser.set_defaults(time_use_amp=True, time_allow_tf32=True)
    parser.add_argument("--time_no_use_amp", dest="time_use_amp", action="store_false")
    parser.add_argument("--time_no_allow_tf32", dest="time_allow_tf32", action="store_false")

    parser.add_argument("--structure_batch_size", type=int, default=4096)
    parser.add_argument("--structure_max_events_in_single_batch", type=int, default=20000)
    parser.add_argument("--structure_close_update_backward", action="store_true", default=False)
    parser.add_argument("--structure_dict_mode", choices=("tag_sum", "tag_max", "per_rel"), default="tag_sum")
    parser.add_argument("--structure_shared_w", choices=("dual_msim", "cross_msim", "unweighted"), default="dual_msim")
    parser.add_argument("--structure_per_rel_use_mtrans", action="store_true", default=False)
    parser.add_argument("--structure_ppr_k", type=int, default=1000)
    parser.add_argument("--structure_top_k_relation", type=int, default=0)
    parser.add_argument("--structure_ppr_alpha", type=float, default=0.01)
    parser.add_argument("--structure_ppr_beta", type=float, default=0.9)
    parser.add_argument("--structure_gamma", type=float, default=0.0)
    parser.add_argument("--structure_direct_single_hop", type=float, default=1.0)
    parser.add_argument("--structure_decay_direct", type=float, default=0.01)
    parser.add_argument("--structure_top_share", type=int, default=100)
    parser.add_argument("--structure_top_direct", type=int, default=-1)
    parser.add_argument("--structure_decay_rel_trans", type=float, default=0.01)
    parser.add_argument("--structure_window_semantic_sim", type=float, default=365.0)
    parser.add_argument("--structure_window_trans", type=float, default=365.0)

    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--source_join_threads", type=int, default=60)
    parser.add_argument("--source_join_log_batches", type=int, default=0)
    parser.add_argument("--dsh_log_bucket_stats", action="store_true", default=False)
    parser.add_argument("--focus_metric", default="H10")
    parser.add_argument("--hybrid_preset_indices", default="")
    parser.add_argument("--hybrid_select_split", choices=("val", "test"), default="test")
    parser.add_argument("--component_splits", default="")
    parser.add_argument("--skip_time_score_check", action="store_true", default=False)
    parser.add_argument("--skip_component_metrics", action="store_true", default=False)
    parser.add_argument("--skip_save_top10", action="store_true", default=False)
    parser.add_argument("--max_hybrid_train_queries", type=int, default=0)
    parser.add_argument("--hybrid_train_query_stride", type=int, default=1)
    parser.add_argument("--rescue_topk", type=int, default=100)
    parser.add_argument("--rescue_min_pos_rank", type=int, default=1)
    parser.add_argument("--rescue_max_pos_rank", type=int, default=100)
    parser.add_argument("--rescue_exclude_top10", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=60)
    parser.add_argument("--b_mode", choices=("continuous", "binary"), default="continuous")
    parser.add_argument("--b_binary_unseen", type=float, default=0.0)
    parser.add_argument("--b_continuous_alpha", type=float, default=0.0001)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
