import argparse
import hashlib
import json
import os
import os.path as osp
from types import SimpleNamespace

import numpy as np

import train_new_structure as tns
import train_time
from new_single_pipeline import time as time_module
from utils import (
    ScoreStore,
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    dense_rank,
    finalize_metric_sums,
    format_bytes,
    load_datasets,
    save_config,
    save_metrics,
    set_random_seed,
)


PROTOCOL = "hybrid_args_lgbm_v1"
SCORE_SUFFIXES = ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json")


def stable_hash(payload, length=12):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[: int(length)]


def require_score_files(out_dir, label, splits=("train", "val", "test")):
    missing = []
    for split in splits:
        for suffix in SCORE_SUFFIXES:
            path = osp.join(out_dir, f"{split}_{suffix}")
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(f"{label} score files are incomplete; first missing: {missing[0]}")


def resolve_time_root(out_dir, time_root):
    if not time_root:
        return out_dir
    rel = out_dir.replace("\\", "/")
    parts = rel.split("/")
    if parts and parts[0] == "results_time_tkg_single":
        rel = "/".join(parts[1:])
    return osp.join(time_root, rel)


def minmax_by_query(scores, valid):
    low = np.min(np.where(valid, scores, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(valid, scores, -np.inf), axis=1, keepdims=True)
    denom = np.maximum(high - low, 1e-12)
    return np.where(valid, (scores - low) / denom, 0.0).astype(np.float32, copy=False)


def zscore_by_query(scores, valid):
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(valid, scores, 0.0), axis=1, keepdims=True) / count
    var = np.sum(np.where(valid, (scores - mean) ** 2, 0.0), axis=1, keepdims=True) / count
    std = np.where(var <= 1e-12, 1.0, np.sqrt(var))
    return np.where(valid, (scores - mean) / std, 0.0).astype(np.float32, copy=False)


def current_rss_bytes():
    try:
        if os.name == "posix" and osp.exists("/proc/self/statm"):
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None
    return None


def feature_names():
    names = []
    for prefix in ("time", "structure", "base"):
        names.extend(
            [
                f"{prefix}_score",
                f"{prefix}_rank",
                f"{prefix}_minmax",
                f"{prefix}_zscore",
                f"{prefix}_top10",
                f"{prefix}_top50",
                f"{prefix}_top100",
            ]
        )
    names.extend(
        [
            "time_minus_structure",
            "time_plus_structure",
            "time_times_structure",
            "rel_id",
            "src_id",
            "candidate_id",
            "candidate_is_source",
        ]
    )
    return names


FEATURE_NAMES = feature_names()


def make_feature_cube(batch_data, cand_ids, valid, time_scores, structure_scores):
    base_scores = ((minmax_by_query(time_scores, valid) + minmax_by_query(structure_scores, valid)) * 0.5).astype(
        np.float32, copy=False
    )
    score_map = {
        "time": np.where(valid, time_scores, 0.0).astype(np.float32, copy=False),
        "structure": np.where(valid, structure_scores, 0.0).astype(np.float32, copy=False),
        "base": np.where(valid, base_scores, 0.0).astype(np.float32, copy=False),
    }
    features = []
    for prefix in ("time", "structure", "base"):
        scores = score_map[prefix]
        ranks = dense_rank(scores, valid).astype(np.float32)
        features.extend(
            [
                scores,
                ranks,
                minmax_by_query(scores, valid),
                zscore_by_query(scores, valid),
                ((ranks <= 10) & valid).astype(np.float32),
                ((ranks <= 50) & valid).astype(np.float32),
                ((ranks <= 100) & valid).astype(np.float32),
            ]
        )
    features.extend(
        [
            score_map["time"] - score_map["structure"],
            score_map["time"] + score_map["structure"],
            score_map["time"] * score_map["structure"],
        ]
    )
    width = valid.shape[1]
    rels = batch_data[:, 1].astype(np.float32).reshape(-1, 1).repeat(width, axis=1)
    src = batch_data[:, 0].astype(np.float32).reshape(-1, 1).repeat(width, axis=1)
    features.extend(
        [
            rels,
            src,
            np.where(valid, cand_ids, 0).astype(np.float32, copy=False),
            (cand_ids == batch_data[:, 0:1]).astype(np.float32, copy=False),
        ]
    )
    cube = np.stack(features, axis=2).astype(np.float32, copy=False)
    bad = int(np.size(cube) - np.sum(np.isfinite(cube)))
    if bad:
        raise RuntimeError(f"hybrid feature cube has {bad} non-finite values")
    return cube, score_map["base"]


def build_time_args(args):
    values = {
        "dataset": args.dataset,
        "seed": args.seed,
        "gpu": 0,
        "output_dir": "",
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


def build_structure_args(args):
    values = {
        "dataset": args.dataset,
        "seed": args.seed,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "batch_size": args.structure_batch_size,
        "max_events_in_single_batch": args.structure_max_events_in_single_batch,
        "source_join_threads": args.structure_source_join_threads,
        "source_join_log_batches": 0,
        "dsh_log_bucket_stats": False,
        "close_update_backward": args.structure_close_update_backward,
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
        "output_root": args.structure_output_root,
    }
    return tns.normalize_args(SimpleNamespace(**values))


def locate_inputs(args):
    time_args = build_time_args(args)
    structure_args = build_structure_args(args)
    time_dir = resolve_time_root(time_module.get_out_dir(time_args), args.time_root)
    structure_dir = tns.make_new_result_dir(structure_args)
    require_score_files(time_dir, "time")
    require_score_files(structure_dir, "structure")
    return time_args, structure_args, time_dir, structure_dir


def iter_score_blocks(data, split, args, time_store, structure_store):
    row_offset = 0
    for events, _, t_orig in split_snapshots(data, split):
        for batch_data, neg_arr, neg_mask in collect_eval_batch(
            events, t_orig, data["negative_sampler"], split, args.query_batch_size
        ):
            width = int(neg_arr.shape[1])
            end = row_offset + len(batch_data)
            time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
            struct_pos, struct_neg, struct_mask = structure_store.get_block(row_offset, end, width)
            valid = neg_mask[:, :width] & time_mask & struct_mask
            all_valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), valid), axis=1)
            cand_ids = np.concatenate((batch_data[:, 2:3], neg_arr[:, :width]), axis=1)
            cand_ids = np.where(all_valid, cand_ids, -1).astype(np.int64, copy=False)
            time_scores = np.concatenate((time_pos, time_neg), axis=1).astype(np.float32, copy=False)
            struct_scores = np.concatenate((struct_pos, struct_neg), axis=1).astype(np.float32, copy=False)
            yield batch_data, cand_ids, all_valid, time_scores, struct_scores
            row_offset = end
    if row_offset != time_store.num_rows or row_offset != structure_store.num_rows:
        raise RuntimeError(
            f"{split} row mismatch: stream={row_offset}, time={time_store.num_rows}, structure={structure_store.num_rows}"
        )


def split_snapshots(data, split):
    if split == "train":
        return data["train_list"][data["train_predict_start_idx"] :]
    if split == "val":
        return data["val_list"]
    if split == "test":
        return data["test_list"]
    raise ValueError(split)


def selected_mask(time_scores, structure_scores, base_scores, valid, topk):
    selected = np.zeros_like(valid, dtype=bool)
    selected[:, 0] = True
    if int(topk) < 0:
        selected |= valid
    else:
        k = int(topk) + 1
        selected |= (dense_rank(time_scores, valid) <= k) & valid
        selected |= (dense_rank(structure_scores, valid) <= k) & valid
        selected |= (dense_rank(base_scores, valid) <= k) & valid
    return selected & valid


def build_train_matrix(data, args, time_dir, structure_dir):
    time_store = ScoreStore(time_dir, "train")
    structure_store = ScoreStore(structure_dir, "train")
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0
    for batch_data, cand_ids, valid, time_scores, struct_scores in iter_score_blocks(
        data, "train", args, time_store, structure_store
    ):
        cube, base_scores = make_feature_cube(batch_data, cand_ids, valid, time_scores, struct_scores)
        selected = selected_mask(time_scores, struct_scores, base_scores, valid, args.hybrid_train_topk)
        for row in range(selected.shape[0]):
            if args.max_train_queries > 0 and queries >= int(args.max_train_queries):
                break
            if (queries % int(args.train_query_stride)) != 0:
                queries += 1
                continue
            cols = np.flatnonzero(selected[row])
            if len(cols) <= 1 or 0 not in cols:
                continue
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[int(np.flatnonzero(cols == 0)[0])] = 1.0
            X_parts.append(cube[row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)
        if args.max_train_queries > 0 and queries >= int(args.max_train_queries):
            break
    if not X_parts:
        raise RuntimeError("no hybrid training rows were built")
    X = np.vstack(X_parts).astype(np.float32, copy=False)
    y = np.concatenate(y_parts).astype(np.float32, copy=False)
    group = np.asarray(groups, dtype=np.int32)
    print(
        f"[Hybrid] train matrix queries={len(group)} rows={len(y)} features={X.shape[1]} "
        f"rss={format_bytes(current_rss_bytes())}",
        flush=True,
    )
    return X, y, group


def validate_lgbm_args(args):
    if int(args.lgbm_n_estimators) <= 0:
        raise ValueError("--lgbm_n_estimators must be > 0")
    if float(args.lgbm_learning_rate) <= 0.0:
        raise ValueError("--lgbm_learning_rate must be > 0")
    if int(args.lgbm_num_leaves) <= 1:
        raise ValueError("--lgbm_num_leaves must be > 1")
    if int(args.lgbm_max_depth) == 0 or int(args.lgbm_max_depth) < -1:
        raise ValueError("--lgbm_max_depth must be -1 or a positive integer")
    if int(args.lgbm_min_child_samples) <= 0:
        raise ValueError("--lgbm_min_child_samples must be > 0")
    if float(args.lgbm_reg_lambda) < 0.0 or float(args.lgbm_reg_alpha) < 0.0:
        raise ValueError("--lgbm_reg_lambda and --lgbm_reg_alpha must be >= 0")
    if float(args.lgbm_min_split_gain) < 0.0:
        raise ValueError("--lgbm_min_split_gain must be >= 0")
    if not 0.0 < float(args.lgbm_subsample) <= 1.0:
        raise ValueError("--lgbm_subsample must be in (0, 1]")
    if not 0.0 < float(args.lgbm_colsample_bytree) <= 1.0:
        raise ValueError("--lgbm_colsample_bytree must be in (0, 1]")


def fit_lgbm_ranker(X, y, group, args):
    validate_lgbm_args(args)
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lightgbm is required") from exc
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=int(args.lgbm_n_estimators),
        learning_rate=float(args.lgbm_learning_rate),
        num_leaves=int(args.lgbm_num_leaves),
        max_depth=int(args.lgbm_max_depth),
        min_child_samples=int(args.lgbm_min_child_samples),
        reg_lambda=float(args.lgbm_reg_lambda),
        reg_alpha=float(args.lgbm_reg_alpha),
        min_split_gain=float(args.lgbm_min_split_gain),
        subsample=float(args.lgbm_subsample),
        colsample_bytree=float(args.lgbm_colsample_bytree),
        random_state=int(args.seed),
        n_jobs=int(args.num_threads),
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    model.fit(X, y, group=group.tolist(), feature_name=FEATURE_NAMES)
    return model


def evaluate_split(data, split, args, time_dir, structure_dir, model):
    time_store = ScoreStore(time_dir, split)
    structure_store = ScoreStore(structure_dir, split)
    sums = {}
    query_count = 0
    for batch_data, cand_ids, valid, time_scores, struct_scores in iter_score_blocks(
        data, split, args, time_store, structure_store
    ):
        cube, _ = make_feature_cube(batch_data, cand_ids, valid, time_scores, struct_scores)
        rows = []
        cols = []
        X_parts = []
        for row in range(valid.shape[0]):
            col = np.flatnonzero(valid[row])
            if len(col) == 0:
                continue
            rows.append(np.full(len(col), row, dtype=np.int64))
            cols.append(col.astype(np.int64, copy=False))
            X_parts.append(cube[row, col, :])
        pred_scores = np.full(valid.shape, -np.inf, dtype=np.float32)
        if X_parts:
            flat_rows = np.concatenate(rows)
            flat_cols = np.concatenate(cols)
            X = np.vstack(X_parts).astype(np.float32, copy=False)
            pred = model.predict(X).astype(np.float32, copy=False)
            pred_scores[flat_rows, flat_cols] = pred
        add_metric_sums(sums, compute_ranking_metric_sums(pred_scores[:, :1], pred_scores[:, 1:], valid[:, 1:]))
        query_count += int(valid.shape[0])
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(query_count)
    print(
        f"[Hybrid] {split} strict: MRR={metrics['mrr_strict']:.5f} "
        f"HR@1={metrics['hit@1_strict']:.5f} HR@10={metrics['hit@10_strict']:.5f}",
        flush=True,
    )
    return metrics


def prefix_metrics(split, metrics):
    return {f"{split}_{key}": value for key, value in metrics.items()}


def make_out_dir(args, time_dir, structure_dir):
    payload = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "seed": args.seed,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "time_dir": time_dir,
        "structure_dir": structure_dir,
        "hybrid_train_topk": args.hybrid_train_topk,
        "lgbm": lgbm_params(args),
    }
    h = stable_hash(payload)
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"hybrid_{h}")


def lgbm_params(args):
    return {
        "n_estimators": int(args.lgbm_n_estimators),
        "learning_rate": float(args.lgbm_learning_rate),
        "num_leaves": int(args.lgbm_num_leaves),
        "max_depth": int(args.lgbm_max_depth),
        "min_child_samples": int(args.lgbm_min_child_samples),
        "reg_lambda": float(args.lgbm_reg_lambda),
        "reg_alpha": float(args.lgbm_reg_alpha),
        "min_split_gain": float(args.lgbm_min_split_gain),
        "subsample": float(args.lgbm_subsample),
        "colsample_bytree": float(args.lgbm_colsample_bytree),
    }


def main(args):
    set_random_seed(args.seed)
    time_args, structure_args, time_dir, structure_dir = locate_inputs(args)
    print(f"[Hybrid] time_dir={time_dir}", flush=True)
    print(f"[Hybrid] structure_dir={structure_dir}", flush=True)
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    X_train, y_train, group = build_train_matrix(data, args, time_dir, structure_dir)
    model = fit_lgbm_ranker(X_train, y_train, group, args)
    val_metrics = evaluate_split(data, "val", args, time_dir, structure_dir, model)
    test_metrics = evaluate_split(data, "test", args, time_dir, structure_dir, model)

    out_dir = make_out_dir(args, time_dir, structure_dir)
    os.makedirs(out_dir, exist_ok=True)
    model_path = osp.join(out_dir, "lgbm_model.txt")
    getattr(model, "booster_", model).save_model(model_path)
    metrics = {
        "format": "new_hybrid_scores_v1",
        "protocol": PROTOCOL,
        "time_dir": time_dir,
        "structure_dir": structure_dir,
        "model_path": model_path,
        "feature_names": FEATURE_NAMES,
        "time_config": vars(time_args).copy(),
        "structure_config": vars(structure_args).copy(),
        "lgbm_params": lgbm_params(args),
    }
    metrics.update(prefix_metrics("val", val_metrics))
    metrics.update(prefix_metrics("test", test_metrics))
    save_config(out_dir, vars(args).copy())
    save_metrics(out_dir, metrics)
    print(f"[Hybrid] output_dir={out_dir}", flush=True)
    return metrics


def add_common_args(parser):
    parser.add_argument("--dataset", default="Yelp-BOI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--query_batch_size", type=int, default=512)
    parser.add_argument("--output_root", default="results_new_hybrid_save_top10")
    parser.add_argument("--time_root", default="")
    parser.add_argument("--structure_output_root", default="results_new_structure")
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--hybrid_train_topk", type=int, default=200)
    parser.add_argument("--max_train_queries", type=int, default=0)
    parser.add_argument("--train_query_stride", type=int, default=1)


def add_time_args(parser):
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
    parser.add_argument("--time_selection_metric", choices=("mrr", "hit10"), default="hit10")
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


def add_structure_args(parser):
    parser.add_argument("--structure_batch_size", type=int, default=None)
    parser.add_argument("--structure_max_events_in_single_batch", type=int, default=None)
    parser.add_argument("--structure_source_join_threads", type=int, default=0)
    parser.add_argument("--structure_close_update_backward", action="store_true", default=False)
    parser.add_argument("--structure_dict_mode", choices=("tag_sum", "tag_max", "per_rel"), default=None)
    parser.add_argument("--structure_shared_w", choices=("dual_msim", "cross_msim", "unweighted"), default=None)
    parser.add_argument("--structure_per_rel_use_mtrans", action="store_true", default=False)
    parser.add_argument("--structure_ppr_k", type=int, default=None)
    parser.add_argument("--structure_top_k_relation", type=int, default=None)
    parser.add_argument("--structure_ppr_alpha", type=float, default=None)
    parser.add_argument("--structure_ppr_beta", type=float, default=None)
    parser.add_argument("--structure_gamma", type=float, default=None)
    parser.add_argument("--structure_direct_single_hop", type=float, default=None)
    parser.add_argument("--structure_decay_direct", type=float, default=None)
    parser.add_argument("--structure_top_share", type=int, default=None)
    parser.add_argument("--structure_top_direct", type=int, default=None)
    parser.add_argument("--structure_decay_rel_trans", type=float, default=None)
    parser.add_argument("--structure_window_semantic_sim", type=float, default=None)
    parser.add_argument("--structure_window_trans", type=float, default=None)


def add_lgbm_args(parser):
    parser.add_argument("--lgbm_n_estimators", type=int, default=64)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.03)
    parser.add_argument("--lgbm_num_leaves", type=int, default=31)
    parser.add_argument("--lgbm_max_depth", type=int, default=8)
    parser.add_argument("--lgbm_min_child_samples", type=int, default=100)
    parser.add_argument("--lgbm_reg_lambda", type=float, default=1.0)
    parser.add_argument("--lgbm_reg_alpha", type=float, default=0.01)
    parser.add_argument("--lgbm_min_split_gain", type=float, default=0.0)
    parser.add_argument("--lgbm_subsample", type=float, default=0.9)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.9)


def build_parser():
    parser = argparse.ArgumentParser("Train a LightGBM hybrid reranker from existing time and structure scores.")
    add_common_args(parser)
    add_time_args(parser)
    add_structure_args(parser)
    add_lgbm_args(parser)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
