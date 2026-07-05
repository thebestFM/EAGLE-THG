import argparse
import gc
import hashlib
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np

from utils import (
    ScoreStore,
    add_rank_sums,
    collect_eval_batch,
    dense_rank,
    describe_loaded_data,
    finalize_metric_sums,
    load_datasets,
    save_config,
    save_metrics,
    set_random_seed,
)


PROTOCOL = "hybrid_simplified_no_recurrence_v1"
COMPONENT_SCORE_NAMES = ("dsh", "dmh", "shared", "direct", "structure_raw")
FEATURE_SUFFIXES = ("score", "z", "minmax", "rank_log", "rank_recip", "top10", "top50", "top100")
SCORE_PREFIXES = ("structure_raw", "time", "dsh", "dmh", "direct", "shared", "base", "time_structure_base")
CROSS_FEATURE_NAMES = (
    "structure_minus_time",
    "direct_minus_time",
    "dsh_minus_time",
    "dmh_minus_time",
    "shared_minus_time",
    "abs_structure_minus_time",
    "structure_times_time",
    "direct_times_time",
    "rank_min_structure_time",
    "rank_gap_structure_time",
    "rank_gap_direct_time",
    "structure_and_time_top10",
    "structure_or_time_top10",
    "structure_and_time_top50",
    "structure_or_time_top50",
)
META_FEATURE_NAMES = ("relation_is_inverse", "candidate_is_source")
EPS = 1e-12


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def stable_hash(payload, length=12):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[: int(length)]


def split_snapshots(data, split):
    if split == "train":
        return data["train_list"][data["train_predict_start_idx"] :]
    if split == "val":
        return data["val_list"]
    if split == "test":
        return data["test_list"]
    raise ValueError(f"unknown split: {split}")


def format_metrics(metrics):
    return (
        f"mrr={metrics['mrr_strict']:.5f} "
        f"hr1={metrics['hit@1_strict']:.5f} "
        f"hr10={metrics['hit@10_strict']:.5f}"
    )


def require_flat_score_store(root, label, splits):
    missing = []
    for split in splits:
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
            path = osp.join(root, f"{split}_{suffix}")
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(f"{label} missing score file, first missing: {missing[0]}")


def require_component_score_store(root, splits):
    missing = []
    for split in splits:
        for name in COMPONENT_SCORE_NAMES:
            for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
                path = osp.join(root, split, name, f"{split}_{suffix}")
                if not osp.isfile(path):
                    missing.append(path)
    if missing:
        raise FileNotFoundError(
            "simplified hybrid needs structure component score stores; "
            f"first missing: {missing[0]}"
        )


def has_component_score_store(root, splits):
    try:
        require_component_score_store(root, splits)
        return True
    except FileNotFoundError:
        return False


def infer_component_dir(args):
    if args.component_dir:
        return args.component_dir
    candidate = args.structure_dir
    if candidate and osp.isdir(osp.join(candidate, "train", "dsh")):
        return candidate
    return osp.join(
        args.component_output_root,
        args.dataset,
        f"seed{args.seed}",
        "component_scores",
        str(args.structure_id or "structure_args"),
    )


def make_structure_args(args):
    payload = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "batch_size": int(args.structure_batch_size),
        "query_batch_size": int(args.query_batch_size),
        "max_events_in_single_batch": int(args.structure_max_events_in_single_batch),
        "dict_mode": args.structure_dict_mode,
        "shared_w": args.structure_shared_w,
        "per_rel_use_mtrans": bool(args.structure_per_rel_use_mtrans),
        "ppr_k": int(args.structure_ppr_k),
        "top_k_relation": int(args.structure_top_k_relation),
        "ppr_alpha": float(args.structure_ppr_alpha),
        "ppr_beta": float(args.structure_ppr_beta),
        "gamma": float(args.structure_gamma),
        "direct_single_hop": float(args.structure_direct_single_hop),
        "decay_direct": float(args.structure_decay_direct),
        "top_share": int(args.structure_top_share),
        "top_direct": int(args.structure_top_direct),
        "decay_rel_trans": float(args.structure_decay_rel_trans),
        "window_semantic_sim": float(args.structure_window_semantic_sim),
        "window_trans": float(args.structure_window_trans),
        "close_update_backward": bool(args.structure_close_update_backward),
        "output_root": args.structure_output_root,
        "source_join_threads": int(args.source_join_threads),
        "source_join_log_batches": int(args.source_join_log_batches),
        "dsh_log_bucket_stats": bool(args.dsh_log_bucket_stats),
    }
    from new_single_pipeline.structure_lgbm import BConfig

    payload["b_cfg"] = BConfig(
        mode=args.b_mode,
        binary_unseen=float(args.b_binary_unseen),
        continuous_alpha=float(args.b_continuous_alpha),
    )
    sargs = SimpleNamespace(**payload)
    import train_new_structure as tns

    if hasattr(tns, "normalize_args"):
        tns.normalize_args(sargs)
    return sargs


def maybe_build_component_scores(args, data, splits):
    component_dir = infer_component_dir(args)
    if has_component_score_store(component_dir, splits):
        print(f"[HybridSimplified][components] cache hit -> {component_dir}", flush=True)
        return component_dir
    if args.component_dir or (args.structure_dir and osp.isdir(osp.join(args.structure_dir, "train", "dsh"))):
        require_component_score_store(component_dir, splits)
        return component_dir
    print(
        f"[HybridSimplified][components] missing cache; generating structure component scores -> {component_dir}",
        flush=True,
    )
    from new_single_pipeline.structure_lgbm import save_component_score_stores

    component_parent = osp.dirname(osp.dirname(component_dir))
    struct_id = osp.basename(osp.normpath(component_dir))
    save_component_score_stores(
        data,
        make_structure_args(args),
        "cpu",
        component_parent,
        struct_id,
        splits=splits,
        compute_metrics=not bool(args.skip_component_metrics),
    )
    require_component_score_store(component_dir, splits)
    return component_dir


def zscore_by_query(scores, valid):
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(valid, scores, 0.0), axis=1, keepdims=True) / count
    var = np.sum(np.where(valid, (scores - mean) ** 2.0, 0.0), axis=1, keepdims=True) / count
    std = np.sqrt(np.maximum(var, EPS))
    return np.where(valid, (scores - mean) / std, 0.0).astype(np.float32, copy=False)


def minmax_by_query(scores, valid):
    low = np.min(np.where(valid, scores, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(valid, scores, -np.inf), axis=1, keepdims=True)
    denom = np.maximum(high - low, EPS)
    out = (scores - low) / denom
    return np.where(valid, out, 0.0).astype(np.float32, copy=False)


def max_norm_by_query(scores, valid):
    high = np.max(np.where(valid, scores, 0.0), axis=1, keepdims=True)
    denom = np.maximum(high, EPS)
    return np.where(valid, scores / denom, 0.0).astype(np.float32, copy=False)


def no_recurrence_base_scores(scores, valid):
    parts = [
        max_norm_by_query(scores["dsh"], valid),
        max_norm_by_query(scores["dmh"], valid),
        max_norm_by_query(scores["shared"], valid),
    ]
    return (sum(parts) / float(len(parts))).astype(np.float32, copy=False)


def load_score_matrix(store, row_offset, rows, width):
    end = row_offset + rows
    pos, neg, mask = store.get_block(row_offset, end, width)
    scores = np.concatenate((pos, neg), axis=1).astype(np.float32, copy=False)
    valid = np.concatenate((np.ones((rows, 1), dtype=bool), mask), axis=1)
    return scores, valid


def prefixed_feature_names(prefixes):
    out = []
    for prefix in prefixes:
        out.extend(f"{prefix}_{suffix}" for suffix in FEATURE_SUFFIXES)
    return out


def structure_cross_feature_names():
    return [
        "structure_minus_time",
        "abs_structure_minus_time",
        "structure_times_time",
        "rank_min_structure_time",
        "rank_gap_structure_time",
        "structure_and_time_top10",
        "structure_or_time_top10",
        "structure_and_time_top50",
        "structure_or_time_top50",
    ]


def ablation_remove_names(group):
    group = str(group or "none").strip().lower().replace("-", "_")
    if group in ("", "none", "full"):
        return set()
    if group == "time":
        return {name for name in prefixed_feature_names(SCORE_PREFIXES) + list(CROSS_FEATURE_NAMES) if "time" in name}
    if group == "direct":
        remove = set(prefixed_feature_names(("dsh", "dmh", "direct", "structure_raw", "base", "time_structure_base")))
        remove.update(
            [
                "direct_minus_time",
                "dsh_minus_time",
                "dmh_minus_time",
                "direct_times_time",
                "rank_gap_direct_time",
            ]
        )
        remove.update(structure_cross_feature_names())
        return remove
    if group == "shared":
        remove = set(prefixed_feature_names(("shared", "structure_raw", "base", "time_structure_base")))
        remove.add("shared_minus_time")
        remove.update(structure_cross_feature_names())
        return remove
    if group == "structure":
        remove = set(prefixed_feature_names(("structure_raw", "time_structure_base")))
        remove.update(structure_cross_feature_names())
        return remove
    if group == "cross":
        return set(CROSS_FEATURE_NAMES)
    if group == "meta":
        return set(META_FEATURE_NAMES)
    raise ValueError(f"unknown ablation group: {group}")


class SimplifiedFeatureBuilder:
    def __init__(self, num_rels, ablation_group="none"):
        self.num_rels = int(num_rels)
        self.ablation_group = str(ablation_group or "none").strip().lower().replace("-", "_")
        self.full_feature_names = []
        self._init_names()
        remove = ablation_remove_names(self.ablation_group)
        known = set(self.full_feature_names)
        unknown = sorted(remove - known)
        if unknown:
            raise RuntimeError(f"ablation {self.ablation_group} contains unknown features: {unknown}")
        self.removed_feature_names = [name for name in self.full_feature_names if name in remove]
        self.keep_feature_mask = np.asarray([name not in remove for name in self.full_feature_names], dtype=bool)
        self.keep_feature_indices = np.flatnonzero(self.keep_feature_mask)
        self.feature_names = [name for name in self.full_feature_names if name not in remove]
        if not self.feature_names:
            raise ValueError(f"ablation {self.ablation_group} removed all features")

    def _add(self, name):
        self.full_feature_names.append(name)

    def _add_score_prefix(self, prefix):
        for suffix in FEATURE_SUFFIXES:
            self._add(f"{prefix}_{suffix}")

    def _init_names(self):
        for prefix in SCORE_PREFIXES:
            self._add_score_prefix(prefix)
        for name in CROSS_FEATURE_NAMES + META_FEATURE_NAMES:
            self._add(name)

    def make(self, scores, time_scores, valid, batch_data, cand_ids):
        score_map = {
            "structure_raw": scores["structure_raw"],
            "time": time_scores,
            "dsh": scores["dsh"],
            "dmh": scores["dmh"],
            "direct": scores["direct"],
            "shared": scores["shared"],
            "base": scores["base"],
        }
        score_map["time_structure_base"] = (
            (minmax_by_query(score_map["structure_raw"], valid) + minmax_by_query(time_scores, valid)) * 0.5
        ).astype(np.float32, copy=False)
        ranks = {name: dense_rank(value, valid).astype(np.float32, copy=False) for name, value in score_map.items()}

        features = []
        for prefix in SCORE_PREFIXES:
            score = np.where(valid, score_map[prefix], 0.0).astype(np.float32, copy=False)
            rank = ranks[prefix]
            features.extend(
                [
                    score,
                    zscore_by_query(score, valid),
                    minmax_by_query(score, valid),
                    np.log1p(rank).astype(np.float32, copy=False),
                    (1.0 / np.maximum(rank, 1.0)).astype(np.float32, copy=False),
                    ((rank <= 10) & valid).astype(np.float32),
                    ((rank <= 50) & valid).astype(np.float32),
                    ((rank <= 100) & valid).astype(np.float32),
                ]
            )

        structure = score_map["structure_raw"]
        time_score = score_map["time"]
        direct = score_map["direct"]
        dsh = score_map["dsh"]
        dmh = score_map["dmh"]
        shared = score_map["shared"]
        sr = ranks["structure_raw"]
        tr = ranks["time"]
        dr = ranks["direct"]
        features.extend(
            [
                (structure - time_score).astype(np.float32, copy=False),
                (direct - time_score).astype(np.float32, copy=False),
                (dsh - time_score).astype(np.float32, copy=False),
                (dmh - time_score).astype(np.float32, copy=False),
                (shared - time_score).astype(np.float32, copy=False),
                np.abs(structure - time_score).astype(np.float32, copy=False),
                (structure * time_score).astype(np.float32, copy=False),
                (direct * time_score).astype(np.float32, copy=False),
                np.minimum(sr, tr).astype(np.float32, copy=False),
                np.abs(sr - tr).astype(np.float32, copy=False),
                np.abs(dr - tr).astype(np.float32, copy=False),
                ((sr <= 10) & (tr <= 10) & valid).astype(np.float32),
                (((sr <= 10) | (tr <= 10)) & valid).astype(np.float32),
                ((sr <= 50) & (tr <= 50) & valid).astype(np.float32),
                (((sr <= 50) | (tr <= 50)) & valid).astype(np.float32),
            ]
        )
        rels = batch_data[:, 1].astype(np.int64, copy=False)
        sources = batch_data[:, 0].astype(np.int64, copy=False)
        features.append((rels.reshape(-1, 1) >= self.num_rels // 2).repeat(valid.shape[1], axis=1).astype(np.float32))
        features.append((cand_ids == sources.reshape(-1, 1)).astype(np.float32, copy=False))
        cube = np.stack(features, axis=2).astype(np.float32, copy=False)
        if cube.shape[2] != len(self.full_feature_names):
            raise RuntimeError(f"feature count mismatch: cube={cube.shape[2]} names={len(self.full_feature_names)}")
        if len(self.keep_feature_indices) != len(self.full_feature_names):
            cube = cube[:, :, self.keep_feature_indices]
        bad = int(np.size(cube) - np.sum(np.isfinite(cube)))
        if bad:
            raise RuntimeError(f"simplified hybrid feature cube has {bad} non-finite values")
        return cube


def open_component_stores(component_dir, split):
    stores = {
        name: ScoreStore(osp.join(component_dir, split, name), split)
        for name in COMPONENT_SCORE_NAMES
    }
    rows = next(iter(stores.values())).num_rows
    for name, store in stores.items():
        if store.num_rows != rows:
            raise RuntimeError(f"component row mismatch for {split}/{name}: {store.num_rows} != {rows}")
    return stores, rows


def iter_blocks(data, split, args, feature_builder, time_dir, component_dir):
    time_store = ScoreStore(time_dir, split)
    component_stores, expected_rows = open_component_stores(component_dir, split)
    if time_store.num_rows != expected_rows:
        raise RuntimeError(f"time/component row mismatch for {split}: time={time_store.num_rows}, component={expected_rows}")

    neg_sampler = data["negative_sampler"]
    row_offset = 0
    for events, _, t_orig in split_snapshots(data, split):
        for batch_data, neg_arr, neg_mask in collect_eval_batch(events, t_orig, neg_sampler, split, args.query_batch_size):
            width = int(neg_arr.shape[1])
            rows = len(batch_data)
            valid = np.concatenate((np.ones((rows, 1), dtype=bool), neg_mask[:, :width]), axis=1)
            scores = {}
            for name, store in component_stores.items():
                values, mask = load_score_matrix(store, row_offset, rows, width)
                valid[:, 1:] &= mask[:, 1:]
                scores[name] = values
            time_scores, time_valid = load_score_matrix(time_store, row_offset, rows, width)
            valid[:, 1:] &= time_valid[:, 1:]
            for name in scores:
                scores[name] = np.where(valid, scores[name], 0.0).astype(np.float32, copy=False)
            scores["base"] = no_recurrence_base_scores(scores, valid)
            time_scores = np.where(valid, time_scores, 0.0).astype(np.float32, copy=False)
            cand_ids = np.concatenate((batch_data[:, 2:3], neg_arr[:, :width]), axis=1)
            cand_ids = np.where(valid, cand_ids, -1).astype(np.int64, copy=False)
            features = feature_builder.make(scores, time_scores, valid, batch_data, cand_ids)
            yield {
                "batch_data": batch_data,
                "valid": valid,
                "scores": scores,
                "time_scores": time_scores,
                "cand_ids": cand_ids,
                "features": features,
            }
            row_offset += rows
    if row_offset != expected_rows:
        raise RuntimeError(f"row count mismatch for {split}: stream={row_offset}, stores={expected_rows}")


def rescue_topk_mask(structure_scores, valid, topk):
    ranks = dense_rank(structure_scores, valid)
    return (ranks <= int(topk)) & valid, ranks


def build_train_matrix(data, args, feature_builder, time_dir, component_dir):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0
    skipped_pos_after_topk = 0
    preserve_queries = 0
    rescue_queries = 0
    include_top10 = not bool(args.rescue_exclude_top10)
    max_queries = int(args.max_hybrid_train_queries or 0)
    query_stride = max(1, int(args.hybrid_train_query_stride or 1))
    eligible_queries = 0
    sampled_out_queries = 0

    for block in iter_blocks(data, "train", args, feature_builder, time_dir, component_dir):
        selected, ranks = rescue_topk_mask(block["scores"]["structure_raw"], block["valid"], args.rescue_topk)
        pos_ranks = ranks[:, 0]
        for row in range(selected.shape[0]):
            if max_queries > 0 and queries >= max_queries:
                break
            pos_rank = int(pos_ranks[row])
            if pos_rank > int(args.rescue_max_pos_rank):
                skipped_pos_after_topk += 1
                continue
            if pos_rank < int(args.rescue_min_pos_rank) and not include_top10:
                continue
            cols = np.flatnonzero(selected[row])
            if len(cols) <= 1 or 0 not in cols:
                skipped_pos_after_topk += 1
                continue
            eligible_queries += 1
            if (eligible_queries - 1) % query_stride != 0:
                sampled_out_queries += 1
                continue
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[int(np.flatnonzero(cols == 0)[0])] = 1.0
            X_parts.append(block["features"][row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)
            if pos_rank <= 10:
                preserve_queries += 1
            else:
                rescue_queries += 1
        if max_queries > 0 and queries >= max_queries:
            break
    if not X_parts:
        raise RuntimeError("no simplified hybrid rows built for train split")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {
            "split": "train",
            "queries": int(queries),
            "rows": int(rows),
            "topk": int(args.rescue_topk),
            "min_pos_rank": int(args.rescue_min_pos_rank),
            "max_pos_rank": int(args.rescue_max_pos_rank),
            "include_top10": bool(include_top10),
            "preserve_queries": int(preserve_queries),
            "rescue_queries": int(rescue_queries),
            "skipped_pos_after_topk": int(skipped_pos_after_topk),
            "eligible_queries": int(eligible_queries),
            "sampled_out_queries": int(sampled_out_queries),
        },
    )


def fit_lgbm_ranker(X, y, group, feature_names, args):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lightgbm is required") from exc
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        max_depth=int(args.max_depth),
        min_child_samples=int(args.min_child_samples),
        reg_lambda=float(args.reg_lambda),
        reg_alpha=float(args.reg_alpha),
        min_split_gain=float(args.min_split_gain),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.seed),
        n_jobs=int(args.num_threads),
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    model.fit(X, y, group=group.tolist(), feature_name=feature_names)
    return model


def predict_lgbm(model, X):
    pred = model.predict(X).astype(np.float32, copy=False)
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("LGBM produced non-finite predictions")
    return pred


def score_rescue_selected(raw_scores, selected, pred):
    final = raw_scores.astype(np.float32, copy=True)
    if not np.any(selected):
        return final
    outside = (~selected) & np.isfinite(final)
    floor = float(np.max(final[outside])) if np.any(outside) else 0.0
    selected_cols = np.flatnonzero(selected)
    raw_sel = raw_scores[selected_cols].astype(np.float32, copy=False)
    order = np.lexsort((selected_cols, -raw_sel, -np.asarray(pred, dtype=np.float32)))
    ordered_cols = selected_cols[order]
    n = int(len(ordered_cols))
    final[ordered_cols] = floor + 1.0 + ((n - np.arange(n, dtype=np.float32)) / max(n, 1))
    return final


def evaluate_model(data, split, args, feature_builder, model, time_dir, component_dir):
    sums = {}
    rescue_stats = {
        "pos_in_top10_before": 0,
        "pos_11_100_before": 0,
        "pos_after_top100_before": 0,
        "pos_in_top10_after": 0,
        "rescued_11_100_to_top10": 0,
        "dropped_top10": 0,
    }
    for block in iter_blocks(data, split, args, feature_builder, time_dir, component_dir):
        selected, ranks = rescue_topk_mask(block["scores"]["structure_raw"], block["valid"], args.rescue_topk)
        loose = []
        strict = []
        avg = []
        for row in range(block["valid"].shape[0]):
            sel_cols = np.flatnonzero(selected[row])
            raw_scores = np.where(block["valid"][row], block["scores"]["structure_raw"][row], -np.inf).astype(np.float32)
            before_rank = int(ranks[row, 0])
            if before_rank <= 10:
                rescue_stats["pos_in_top10_before"] += 1
            elif before_rank <= int(args.rescue_topk):
                rescue_stats["pos_11_100_before"] += 1
            else:
                rescue_stats["pos_after_top100_before"] += 1
            if len(sel_cols):
                X = block["features"][row, sel_cols, :].astype(np.float32, copy=False)
                pred = predict_lgbm(model, X)
                final_scores = score_rescue_selected(raw_scores, selected[row], pred)
            else:
                final_scores = raw_scores
            pos_score = final_scores[0]
            neg_scores = final_scores[1:]
            neg_valid = block["valid"][row, 1:]
            l_rank = 1 + int(np.sum((neg_scores > pos_score) & neg_valid))
            s_rank = 1 + int(np.sum((neg_scores >= pos_score) & neg_valid))
            loose.append(l_rank)
            strict.append(s_rank)
            avg.append((l_rank + s_rank) * 0.5)
            if s_rank <= 10:
                rescue_stats["pos_in_top10_after"] += 1
                if 10 < before_rank <= int(args.rescue_topk):
                    rescue_stats["rescued_11_100_to_top10"] += 1
            elif before_rank <= 10:
                rescue_stats["dropped_top10"] += 1
        add_rank_sums(
            sums,
            np.asarray(loose, dtype=np.int64),
            np.asarray(strict, dtype=np.int64),
            np.asarray(avg, dtype=np.float64),
        )
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    metrics["rescue_stats"] = rescue_stats
    return metrics


def evaluate_score_store(out_dir, data, split, args):
    store = ScoreStore(out_dir, split)
    sums = {}
    row_offset = 0
    neg_sampler = data["negative_sampler"]
    for events, _, t_orig in split_snapshots(data, split):
        for batch_data, neg_arr, neg_mask in collect_eval_batch(events, t_orig, neg_sampler, split, args.query_batch_size):
            width = int(neg_arr.shape[1])
            end = row_offset + len(batch_data)
            pos, neg, mask = store.get_block(row_offset, end, width)
            valid_mask = neg_mask[:, :width] & mask
            loose = 1 + np.sum((neg > pos) & valid_mask, axis=1)
            strict = 1 + np.sum((neg >= pos) & valid_mask, axis=1)
            avg = (loose + strict) * 0.5
            add_rank_sums(sums, loose.astype(np.int64), strict.astype(np.int64), avg.astype(np.float64))
            row_offset = end
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


def save_lgbm_model(model, path):
    ensure_dir(osp.dirname(path))
    booster = getattr(model, "booster_", model)
    booster.save_model(path)


def make_out_dir(args, component_dir):
    payload = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "seed": args.seed,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "time_dir": args.time_dir,
        "structure_dir": args.structure_dir,
        "component_dir": component_dir,
        "ablation": args.ablation,
        "rescue_topk": args.rescue_topk,
        "rescue_min_pos_rank": args.rescue_min_pos_rank,
        "rescue_max_pos_rank": args.rescue_max_pos_rank,
        "rescue_exclude_top10": args.rescue_exclude_top10,
        "hybrid_select_split": args.hybrid_select_split,
        "lgbm": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "min_split_gain": args.min_split_gain,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        },
    }
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"simplified_{stable_hash(payload)}")


def validate_args(args):
    args.ablation = str(getattr(args, "ablation", "none") or "none").strip().lower().replace("-", "_")
    ablation_remove_names(args.ablation)
    args.structure_id = str(getattr(args, "structure_id", "") or "structure_args")
    if int(args.query_batch_size) <= 0:
        raise ValueError("--query_batch_size must be > 0")
    if int(args.rescue_topk) <= 0:
        raise ValueError("--rescue_topk must be > 0")
    if int(args.rescue_min_pos_rank) <= 0 or int(args.rescue_max_pos_rank) < int(args.rescue_min_pos_rank):
        raise ValueError("--rescue_min_pos_rank/--rescue_max_pos_rank invalid")
    if int(args.rescue_max_pos_rank) > int(args.rescue_topk):
        raise ValueError("--rescue_max_pos_rank cannot exceed --rescue_topk")
    if str(args.hybrid_select_split) not in ("val", "test"):
        raise ValueError("--hybrid_select_split must be val or test")
    if int(args.num_threads) <= 0:
        raise ValueError("--num_threads must be > 0")


def run(args):
    validate_args(args)
    set_random_seed(args.seed)
    splits = tuple(dict.fromkeys(("train", args.hybrid_select_split, "test")))
    require_flat_score_store(args.time_dir, "time_dir", splits)
    print(
        f"[HybridSimplified] protocol={PROTOCOL} dataset={args.dataset} "
        f"ns_q={args.ns_q} select_split={args.hybrid_select_split} ablation={args.ablation}",
        flush=True,
    )
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[HybridSimplified]")
    component_dir = maybe_build_component_scores(args, data, splits)
    out_dir = ensure_dir(make_out_dir(args, component_dir))
    print(f"[HybridSimplified] output -> {out_dir}", flush=True)
    if args.structure_dir and osp.isdir(args.structure_dir) and osp.isfile(osp.join(args.structure_dir, "test_pos.npy")):
        structure_metrics = evaluate_score_store(args.structure_dir, data, "test", args)
        print(f"[HybridSimplified][structure_raw_store] test {format_metrics(structure_metrics)}", flush=True)
    time_metrics = evaluate_score_store(args.time_dir, data, "test", args)
    print(f"[HybridSimplified][time] test {format_metrics(time_metrics)}", flush=True)

    num_rels = int(data.get("num_rels_raw", data["num_rels"])) * 2 if data.get("is_thg", False) else int(data["num_rels"])
    feature_builder = SimplifiedFeatureBuilder(num_rels, ablation_group=args.ablation)
    print(
        f"[HybridSimplified] features={len(feature_builder.feature_names)}/{len(feature_builder.full_feature_names)} "
        f"removed={len(feature_builder.removed_feature_names)} names={feature_builder.feature_names}",
        flush=True,
    )
    if feature_builder.removed_feature_names:
        print(f"[HybridSimplified][ablation] removed={feature_builder.removed_feature_names}", flush=True)

    t0 = time.time()
    X_train, y_train, group, train_info = build_train_matrix(data, args, feature_builder, args.time_dir, component_dir)
    print(
        f"[HybridSimplified] train rows={train_info['rows']} queries={train_info['queries']} "
        f"features={X_train.shape[1]} preserve={train_info['preserve_queries']} "
        f"rescue={train_info['rescue_queries']}",
        flush=True,
    )
    model = fit_lgbm_ranker(X_train, y_train, group, feature_builder.feature_names, args)
    del X_train, y_train, group
    gc.collect()

    select_metrics = evaluate_model(data, args.hybrid_select_split, args, feature_builder, model, args.time_dir, component_dir)
    test_metrics = select_metrics if args.hybrid_select_split == "test" else evaluate_model(
        data, "test", args, feature_builder, model, args.time_dir, component_dir
    )
    model_path = osp.join(out_dir, "model.txt")
    save_lgbm_model(model, model_path)
    summary = {
        "format": "hybrid_simplified_summary_v1",
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "args": vars(args).copy(),
        "time_dir": args.time_dir,
        "structure_dir": args.structure_dir,
        "component_dir": component_dir,
        "ablation": args.ablation,
        "full_feature_names": feature_builder.full_feature_names,
        "feature_names": feature_builder.feature_names,
        "removed_feature_names": feature_builder.removed_feature_names,
        "keep_feature_mask": feature_builder.keep_feature_mask.astype(int).tolist(),
        "train_info": train_info,
        "selection_split": args.hybrid_select_split,
        "selection_metrics": select_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "elapsed_s": time.time() - t0,
    }
    save_config(out_dir, summary["args"])
    save_metrics(out_dir, summary)
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[HybridSimplified] test {format_metrics(test_metrics)} model={model_path}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("Simplified no-recurrence rescue hybrid from saved score stores.")
    parser.add_argument("--dataset", default="ICEWS14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--time_dir", required=True)
    parser.add_argument("--structure_dir", default="", help="Optional final structure score dir; component dir may also be passed here.")
    parser.add_argument("--component_dir", default="", help="component_scores/<structure_id> dir containing dsh/dmh/shared/direct/structure_raw stores.")
    parser.add_argument("--component_output_root", default="results_hybrid_simplified_components")
    parser.add_argument("--output_root", default="results_hybrid_simplified")
    parser.add_argument(
        "--ablation",
        choices=("none", "time", "direct", "shared", "structure", "cross", "meta"),
        default="none",
        help="Feature group to remove with a name-based mask after building the full simplified feature cube.",
    )
    parser.add_argument("--query_batch_size", type=int, default=512)
    parser.add_argument("--hybrid_select_split", choices=("val", "test"), default="test")
    parser.add_argument("--rescue_topk", type=int, default=100)
    parser.add_argument("--rescue_min_pos_rank", type=int, default=1)
    parser.add_argument("--rescue_max_pos_rank", type=int, default=100)
    parser.add_argument("--rescue_exclude_top10", action="store_true", default=False)
    parser.add_argument("--max_hybrid_train_queries", type=int, default=0)
    parser.add_argument("--hybrid_train_query_stride", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--min_child_samples", type=int, default=80)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.01)
    parser.add_argument("--min_split_gain", type=float, default=0.001)
    parser.add_argument("--subsample", type=float, default=0.95)
    parser.add_argument("--colsample_bytree", type=float, default=0.85)
    parser.add_argument("--structure_id", default="structure_args")
    parser.add_argument("--structure_output_root", default="results_new_structure")
    parser.add_argument("--structure_batch_size", type=int, default=4096)
    parser.add_argument("--structure_max_events_in_single_batch", type=int, default=20000)
    parser.add_argument("--structure_dict_mode", choices=("tag_sum", "tag_max", "per_rel"), default="tag_sum")
    parser.add_argument("--structure_shared_w", choices=("dual_msim", "cross_msim", "unweighted"), default="dual_msim")
    parser.add_argument("--structure_per_rel_use_mtrans", action="store_true", default=False)
    parser.add_argument("--structure_ppr_k", type=int, default=1000)
    parser.add_argument("--structure_top_k_relation", type=int, default=0)
    parser.add_argument("--structure_ppr_alpha", type=float, default=0.03)
    parser.add_argument("--structure_ppr_beta", type=float, default=0.9)
    parser.add_argument("--structure_gamma", type=float, default=0.0)
    parser.add_argument("--structure_direct_single_hop", type=float, default=1.0)
    parser.add_argument("--structure_decay_direct", type=float, default=0.01)
    parser.add_argument("--structure_top_share", type=int, default=100)
    parser.add_argument("--structure_top_direct", type=int, default=-1)
    parser.add_argument("--structure_decay_rel_trans", type=float, default=0.01)
    parser.add_argument("--structure_window_semantic_sim", type=float, default=365.0)
    parser.add_argument("--structure_window_trans", type=float, default=365.0)
    parser.add_argument("--structure_close_update_backward", action="store_true", default=False)
    parser.add_argument("--source_join_threads", type=int, default=60)
    parser.add_argument("--source_join_log_batches", type=int, default=0)
    parser.add_argument("--dsh_log_bucket_stats", action="store_true", default=False)
    parser.add_argument("--b_mode", choices=("binary", "continuous"), default="continuous")
    parser.add_argument("--b_binary_unseen", type=float, default=0.0)
    parser.add_argument("--b_continuous_alpha", type=float, default=1e-4)
    parser.add_argument("--skip_component_metrics", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
