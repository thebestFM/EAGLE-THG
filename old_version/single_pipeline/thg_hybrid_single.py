import argparse
import gc
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from single_pipeline.thg_common import (
    ScoreStore,
    add_rank_sums,
    dense_rank,
    finalize_rank_sums,
    format_metrics,
    load_ranker_matrix_cache,
    minmax_by_query,
    prefix_metrics,
    ranker_matrix_cache_complete,
    ranker_matrix_cache_dir,
    ranks_from_candidate_scores,
    save_ranker_matrix_cache,
    stable_hash,
    tail_query_window,
    zscore_by_query,
)
from single_pipeline.thg_structure_single import StructureFeatureBuilder, iter_feature_blocks
from utils import ScoreWriter, describe_loaded_data, is_run_complete, load_config, load_datasets, load_metrics, save_config, save_metrics, set_random_seed


class HybridFeatureBuilder:
    def __init__(self, structure_feature_names):
        self.structure_feature_names = list(structure_feature_names)
        self.feature_names = []
        self._init_names()

    def _add(self, name):
        self.feature_names.append(name)

    def _init_names(self):
        for prefix in ("time", "structure", "base"):
            self._add(f"{prefix}_score")
            self._add(f"{prefix}_z")
            self._add(f"{prefix}_minmax")
            self._add(f"{prefix}_rank_log")
            self._add(f"{prefix}_rank_recip")
        for name in (
            "time_minus_structure",
            "abs_time_minus_structure",
            "time_times_structure",
            "rank_min",
            "rank_gap",
            "both_top10",
            "either_top10",
            "both_top50",
            "either_top50",
        ):
            self._add(name)
        for name in self.structure_feature_names:
            self._add(f"ppr_{name}")

    def make(self, time_scores, structure_scores, structure_cube, valid):
        base_scores = ((minmax_by_query(time_scores, valid) + minmax_by_query(structure_scores, valid)) * 0.5).astype(np.float32)
        score_map = {"time": time_scores, "structure": structure_scores, "base": base_scores}
        ranks = {name: dense_rank(score, valid) for name, score in score_map.items()}
        features = []
        for prefix in ("time", "structure", "base"):
            score = np.where(valid, score_map[prefix], 0.0).astype(np.float32)
            rank = ranks[prefix].astype(np.float32)
            features.extend(
                [
                    score,
                    zscore_by_query(score, valid),
                    minmax_by_query(score, valid),
                    np.log1p(rank).astype(np.float32),
                    (1.0 / np.maximum(rank, 1.0)).astype(np.float32),
                ]
            )
        time_rank = ranks["time"].astype(np.float32)
        struct_rank = ranks["structure"].astype(np.float32)
        features.extend(
            [
                (time_scores - structure_scores).astype(np.float32),
                np.abs(time_scores - structure_scores).astype(np.float32),
                (time_scores * structure_scores).astype(np.float32),
                np.minimum(time_rank, struct_rank).astype(np.float32),
                np.abs(time_rank - struct_rank).astype(np.float32),
                ((time_rank <= 10) & (struct_rank <= 10)).astype(np.float32),
                ((time_rank <= 10) | (struct_rank <= 10)).astype(np.float32),
                ((time_rank <= 50) & (struct_rank <= 50)).astype(np.float32),
                ((time_rank <= 50) | (struct_rank <= 50)).astype(np.float32),
            ]
        )
        for idx in range(structure_cube.shape[-1]):
            features.append(np.where(valid, structure_cube[:, :, idx], 0.0).astype(np.float32))
        return np.stack(features, axis=2).astype(np.float32)


def load_namespace_config(out_dir):
    cfg = load_config(out_dir)
    return SimpleNamespace(**cfg)


def validate_score_run(out_dir, args, label):
    cfg = load_config(out_dir)
    for key in ("dataset", "seed", "ns_q", "ns_seed", "train_predict_ratio"):
        if str(cfg.get(key)) != str(getattr(args, key)):
            raise ValueError(f"{label} config mismatch for {key}: {out_dir}")
    return cfg


def _expected_modes(eval_test=True):
    modes = ("train", "val")
    if eval_test:
        modes = modes + ("test",)
    return modes


def _require_attrs(args, names, label):
    missing = [name for name in names if not hasattr(args, name)]
    if missing:
        raise ValueError(f"cannot auto-run {label}; missing args: {', '.join(missing)}")


def make_time_dependency_args(args):
    names = (
        "time_topk",
        "time_node_dim",
        "time_rel_dim",
        "time_hidden_dim",
        "time_dropout",
        "time_batch_size",
        "time_eval_batch_size",
        "time_eval_neg_chunk",
        "time_train_num_neg",
        "time_hard_neg_ratio",
        "time_train_loss",
        "time_temperature",
        "time_rank_margin",
        "time_lr",
        "time_weight_decay",
        "time_grad_clip",
        "time_num_epochs",
        "time_patience",
        "time_tolerance",
        "time_selection_metric",
        "gpu",
    )
    _require_attrs(args, names, "time")
    return SimpleNamespace(
        dataset=args.dataset,
        seed=args.seed,
        gpu=args.gpu,
        ns_q=args.ns_q,
        ns_seed=args.ns_seed,
        train_predict_ratio=args.train_predict_ratio,
        topk=args.time_topk,
        node_dim=args.time_node_dim,
        rel_dim=args.time_rel_dim,
        hidden_dim=args.time_hidden_dim,
        dropout=args.time_dropout,
        batch_size=args.time_batch_size,
        eval_batch_size=args.time_eval_batch_size,
        eval_neg_chunk=args.time_eval_neg_chunk,
        train_num_neg=args.time_train_num_neg,
        hard_neg_ratio=args.time_hard_neg_ratio,
        train_loss=args.time_train_loss,
        temperature=args.time_temperature,
        rank_margin=args.time_rank_margin,
        lr=args.time_lr,
        weight_decay=args.time_weight_decay,
        grad_clip=args.time_grad_clip,
        num_epochs=args.time_num_epochs,
        patience=args.time_patience,
        tolerance=args.time_tolerance,
        selection_metric=args.time_selection_metric,
        force=getattr(args, "force_time", False),
        eval_test=getattr(args, "eval_test", True),
    )


def make_structure_dependency_args(args):
    names = (
        "structure_block_size",
        "structure_train_topk",
        "geo_topk",
        "geo_tau_km",
        "user_half_life_days",
        "business_half_life_days",
        "direct_weight",
        "geo_dynamic_weight",
        "structure_val_early_stop_tailk",
        "lgbm_n_estimators",
        "lgbm_learning_rate",
        "lgbm_num_leaves",
        "lgbm_max_depth",
        "lgbm_min_child_samples",
        "lgbm_reg_lambda",
        "lgbm_reg_alpha",
        "lgbm_subsample",
        "lgbm_colsample_bytree",
        "lgbm_early_stopping_rounds",
        "num_threads",
    )
    _require_attrs(args, names, "structure")
    return SimpleNamespace(
        dataset=args.dataset,
        seed=args.seed,
        ns_q=args.ns_q,
        ns_seed=args.ns_seed,
        train_predict_ratio=args.train_predict_ratio,
        block_size=args.structure_block_size,
        train_topk=args.structure_train_topk,
        geo_topk=args.geo_topk,
        geo_tau_km=args.geo_tau_km,
        user_half_life_days=args.user_half_life_days,
        business_half_life_days=args.business_half_life_days,
        direct_weight=args.direct_weight,
        geo_dynamic_weight=args.geo_dynamic_weight,
        val_early_stop_tailk=args.structure_val_early_stop_tailk,
        n_estimators=args.lgbm_n_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        max_depth=args.lgbm_max_depth,
        min_child_samples=args.lgbm_min_child_samples,
        reg_lambda=args.lgbm_reg_lambda,
        reg_alpha=args.lgbm_reg_alpha,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample_bytree,
        early_stopping_rounds=args.lgbm_early_stopping_rounds,
        num_threads=args.num_threads,
        force=getattr(args, "force_structure", False),
        force_feature_cache=getattr(args, "force_feature_cache", False),
        eval_test=getattr(args, "eval_test", True),
    )


def ensure_dependency_runs(args):
    from single_pipeline import thg_structure_single, thg_time_single

    if not hasattr(args, "auto_run_dependencies"):
        args.auto_run_dependencies = False
    if not hasattr(args, "time_dir"):
        args.time_dir = ""
    if not hasattr(args, "structure_dir"):
        args.structure_dir = ""
    if not args.auto_run_dependencies:
        return args

    modes = _expected_modes(getattr(args, "eval_test", True))
    if not args.time_dir:
        time_args = make_time_dependency_args(args)
        args.time_dir = thg_time_single.get_out_dir(time_args)
        print(f"[THG-Hybrid] time dependency -> {args.time_dir}", flush=True)
        thg_time_single.main(time_args)
    elif not is_run_complete(args.time_dir, modes):
        time_args = make_time_dependency_args(args)
        derived = thg_time_single.get_out_dir(time_args)
        if osp.normpath(derived) != osp.normpath(args.time_dir):
            raise ValueError(f"time_dir is incomplete and does not match current dependency args: {args.time_dir}")
        print(f"[THG-Hybrid] rebuilding incomplete time dependency -> {args.time_dir}", flush=True)
        thg_time_single.main(time_args)

    if not args.structure_dir:
        structure_args = make_structure_dependency_args(args)
        args.structure_dir = thg_structure_single.get_out_dir(structure_args)
        print(f"[THG-Hybrid] structure dependency -> {args.structure_dir}", flush=True)
        thg_structure_single.main(structure_args)
    elif not is_run_complete(args.structure_dir, modes):
        structure_args = make_structure_dependency_args(args)
        derived = thg_structure_single.get_out_dir(structure_args)
        if osp.normpath(derived) != osp.normpath(args.structure_dir):
            raise ValueError(f"structure_dir is incomplete and does not match current dependency args: {args.structure_dir}")
        print(f"[THG-Hybrid] rebuilding incomplete structure dependency -> {args.structure_dir}", flush=True)
        thg_structure_single.main(structure_args)
    return args


def iter_hybrid_blocks(data, split, structure_args, feature_builder, args):
    struct_store = ScoreStore(args.structure_dir, split)
    time_store = ScoreStore(args.time_dir, split)
    row_offset = 0
    for batch, struct_valid, _, _, struct_cube in iter_feature_blocks(data, split, StructureFeatureBuilder(), structure_args):
        width = struct_valid.shape[1] - 1
        end = row_offset + len(batch)
        s_pos, s_neg, s_mask = struct_store.get_block(row_offset, end, width)
        t_pos, t_neg, t_mask = time_store.get_block(row_offset, end, width)
        valid = struct_valid & np.concatenate((np.ones((len(batch), 1), dtype=bool), s_mask & t_mask), axis=1)
        structure_scores = np.concatenate((s_pos, s_neg), axis=1).astype(np.float32)
        time_scores = np.concatenate((t_pos, t_neg), axis=1).astype(np.float32)
        structure_scores = np.where(valid, structure_scores, 0.0)
        time_scores = np.where(valid, time_scores, 0.0)
        features = feature_builder.make(time_scores, structure_scores, struct_cube, valid)
        yield valid, time_scores, structure_scores, features
        row_offset = end
    if row_offset != struct_store.num_rows or row_offset != time_store.num_rows:
        raise ValueError(f"score row mismatch for {split}: stream={row_offset}, struct={struct_store.num_rows}, time={time_store.num_rows}")


def train_selection(time_scores, structure_scores, valid, topk):
    time_rank = dense_rank(time_scores, valid)
    struct_rank = dense_rank(structure_scores, valid)
    base = (minmax_by_query(time_scores, valid) + minmax_by_query(structure_scores, valid)) * 0.5
    base_rank = dense_rank(base, valid)
    selected = ((time_rank <= int(topk) + 1) | (struct_rank <= int(topk) + 1) | (base_rank <= int(topk) + 1)) & valid
    selected[:, 0] = True
    return selected


def build_ranker_matrix(data, split, structure_args, feature_builder, args, topk=None, start_query=0, max_queries=None):
    X_parts, y_parts, groups = [], [], []
    queries = 0
    rows = 0
    seen_queries = 0
    done = False
    desc = f"thg_hybrid_build_{split}"
    if start_query or max_queries is not None:
        desc += "_slice"
    for valid, time_scores, structure_scores, features in tqdm(
        iter_hybrid_blocks(data, split, structure_args, feature_builder, args),
        desc=desc,
        leave=False,
    ):
        selected = valid.copy() if topk is None else train_selection(time_scores, structure_scores, valid, topk)
        for i in range(selected.shape[0]):
            if seen_queries < int(start_query):
                seen_queries += 1
                continue
            if max_queries is not None and queries >= int(max_queries):
                done = True
                break
            seen_queries += 1
            cols = np.flatnonzero(selected[i])
            if len(cols) <= 1:
                continue
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[np.flatnonzero(cols == 0)[0]] = 1.0
            X_parts.append(features[i, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)
        if done:
            break
    if not X_parts:
        raise ValueError(f"no hybrid {split} rows built")
    return (
        np.vstack(X_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.float32),
        np.asarray(groups, dtype=np.int32),
        {
            "queries": int(queries),
            "rows": int(rows),
            "start_query": int(start_query),
            "max_queries": None if max_queries is None else int(max_queries),
            "seen_queries": int(seen_queries),
        },
    )


def hybrid_matrix_cache_dir(args, split, topk, start_query, max_queries):
    payload = {
        "format": "thg_hybrid_ranker_matrix_v1",
        "dataset": args.dataset,
        "seed": args.seed,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "split": split,
        "topk": topk,
        "start_query": int(start_query),
        "max_queries": None if max_queries is None else int(max_queries),
        "structure_dir": args.structure_dir,
        "time_dir": args.time_dir,
    }
    return ranker_matrix_cache_dir("cache_thg_hybrid_ranker", args.dataset, args.seed, payload)


def build_or_load_ranker_matrix(data, split, structure_args, feature_builder, args, topk=None, start_query=0, max_queries=None, cache_name=None):
    cache_name = cache_name or split
    cache_dir = hybrid_matrix_cache_dir(args, split, topk, start_query, max_queries)
    if ranker_matrix_cache_complete(cache_dir, cache_name) and not getattr(args, "force_feature_cache", False):
        return load_ranker_matrix_cache(cache_dir, cache_name)
    X, y, group, info = build_ranker_matrix(data, split, structure_args, feature_builder, args, topk, start_query, max_queries)
    info = {**info, "cache_dir": cache_dir, "cache_name": cache_name}
    save_ranker_matrix_cache(cache_dir, cache_name, X, y, group, info)
    print(f"[THG-Hybrid] saved {cache_name} ranker matrix -> {cache_dir}", flush=True)
    return X, y, group, info


def default_lgbm_params(args):
    return {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "max_depth": int(args.max_depth),
        "min_child_samples": int(args.min_child_samples),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(args.reg_alpha),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
    }


def fit_lgbm(X, y, group, X_val, y_val, group_val, feature_builder, args):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("THG hybrid requires lightgbm") from exc
    params = default_lgbm_params(args)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=int(args.seed),
        n_jobs=int(args.num_threads),
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    callbacks = []
    if int(args.early_stopping_rounds) > 0:
        callbacks.append(lgb.early_stopping(int(args.early_stopping_rounds), verbose=False))
    model.fit(
        X,
        y,
        group=group.tolist(),
        feature_name=feature_builder.feature_names,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val.tolist()],
        eval_at=[10],
        callbacks=callbacks,
    )
    return model


def evaluate_model(data, split, structure_args, feature_builder, model, args, out_dir=None, write_scores=False):
    writer = ScoreWriter(out_dir, split) if write_scores else None
    sums = {}
    score_time = 0.0
    start = time.time()
    for valid, _, _, features in tqdm(iter_hybrid_blocks(data, split, structure_args, feature_builder, args), desc=f"thg_hybrid_eval_{split}", leave=False):
        scores = np.zeros(valid.shape, dtype=np.float32)
        flat_valid = valid.reshape(-1)
        if np.any(flat_valid):
            flat_features = features.reshape(-1, features.shape[-1])
            t0 = time.time()
            scores.reshape(-1)[flat_valid] = model.predict(flat_features[flat_valid]).astype(np.float32)
            score_time += time.time() - t0
        loose, strict, avg = ranks_from_candidate_scores(scores, valid)
        add_rank_sums(sums, loose, strict, avg)
        if writer is not None:
            writer.write_batch(scores[:, :1], scores[:, 1:], valid[:, 1:])
    if writer is not None:
        writer.close()
    metrics = finalize_rank_sums(sums)
    metrics["profile"] = {"eval_time_sec": float(time.time() - start), "score_time_sec": float(score_time)}
    print(f"[THG-Hybrid] {split} {format_metrics(metrics)} time={metrics['profile']['eval_time_sec']:.1f}s", flush=True)
    return metrics


def save_lgbm_model(model, path):
    booster = getattr(model, "booster_", model)
    booster.save_model(path)


def get_out_dir(args):
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "structure_dir": args.structure_dir,
        "time_dir": args.time_dir,
        "train_topk": args.train_topk,
        "val_early_stop_tailk": getattr(args, "val_early_stop_tailk", 5000),
        "lgbm": default_lgbm_params(args),
    }
    h = stable_hash(payload, length=12)
    return osp.join(
        "results_thg_hybrid",
        args.dataset,
        f"seed{args.seed}",
        f"r{h}_traink{args.train_topk}_nsq{args.ns_q}_tpr{args.train_predict_ratio:g}",
    )


def validate_args(args):
    if not hasattr(args, "val_early_stop_tailk"):
        args.val_early_stop_tailk = 5000
    if not hasattr(args, "force_feature_cache"):
        args.force_feature_cache = False
    if not args.structure_dir or not args.time_dir:
        raise ValueError("--structure_dir and --time_dir are required or must be resolved before validation")
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or positive")
    if not 0.0 < float(args.train_predict_ratio) < 1.0:
        raise ValueError("--train_predict_ratio must be in (0,1)")
    if int(args.val_early_stop_tailk) < 0:
        raise ValueError("--val_early_stop_tailk must be non-negative")


def main(args):
    ensure_dependency_runs(args)
    validate_args(args)
    set_random_seed(args.seed)
    validate_score_run(args.structure_dir, args, "structure")
    validate_score_run(args.time_dir, args, "time")
    out_dir = get_out_dir(args)
    expected_modes = ("train", "val")
    if getattr(args, "eval_test", True):
        expected_modes = expected_modes + ("test",)
    if is_run_complete(out_dir, expected_modes) and osp.isfile(osp.join(out_dir, "best_lgbm.txt")) and not getattr(args, "force", False):
        print(f"[THG-Hybrid] already complete: {out_dir}", flush=True)
        return load_metrics(out_dir)

    data = load_datasets(args.dataset, q=args.ns_q, load_train_ratio=args.train_predict_ratio, ns_seed=args.ns_seed)
    if not data.get("is_thg"):
        raise ValueError("THG hybrid requires a Yelp THG dataset")
    describe_loaded_data(data, prefix="[THG-Hybrid]")
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()

    structure_args = load_namespace_config(args.structure_dir)
    structure_feature_builder = StructureFeatureBuilder()
    feature_builder = HybridFeatureBuilder(structure_feature_builder.feature_names)
    X_train, y_train, group, train_info = build_or_load_ranker_matrix(data, "train", structure_args, feature_builder, args, topk=args.train_topk)
    val_start, val_count, val_total = tail_query_window(data, "val", args.val_early_stop_tailk)
    X_val, y_val, group_val, val_info = build_or_load_ranker_matrix(
        data,
        "val",
        structure_args,
        feature_builder,
        args,
        topk=None,
        start_query=val_start,
        max_queries=val_count,
        cache_name="val_es",
    )
    print(
        f"[THG-Hybrid] train rows={train_info['rows']} queries={train_info['queries']} "
        f"val_es rows={val_info['rows']} queries={val_info['queries']}/{val_total} "
        f"start={val_start} features={X_train.shape[1]}",
        flush=True,
    )
    model = fit_lgbm(X_train, y_train, group, X_val, y_val, group_val, feature_builder, args)
    del X_train, y_train, group, X_val, y_val, group_val
    gc.collect()

    train_metrics = evaluate_model(data, "train", structure_args, feature_builder, model, args, out_dir, True)
    val_metrics = evaluate_model(data, "val", structure_args, feature_builder, model, args, out_dir, True)
    test_metrics = None
    if getattr(args, "eval_test", True):
        test_metrics = evaluate_model(data, "test", structure_args, feature_builder, model, args, out_dir, True)
    model_path = osp.join(out_dir, "best_lgbm.txt")
    save_lgbm_model(model, model_path)
    metrics = {
        "format": "thg_hybrid_lgbm_v1",
        "model_path": model_path,
        "feature_names": feature_builder.feature_names,
        "train_info": train_info,
        "val_early_stop_info": val_info,
        "runtime_sec": float(time.time() - start),
    }
    metrics.update(prefix_metrics("train", train_metrics))
    metrics.update(prefix_metrics("val", val_metrics))
    if test_metrics is not None:
        metrics.update(prefix_metrics("test", test_metrics))
    save_config(
        out_dir,
        {
            **vars(args),
            "out_dir": out_dir,
            "feature_names": feature_builder.feature_names,
            "structure_feature_names": structure_feature_builder.feature_names,
        },
    )
    save_metrics(out_dir, metrics)
    print(f"[THG-Hybrid] saved -> {out_dir}", flush=True)
    return metrics


def load_args():
    parser = argparse.ArgumentParser("Run the THG hybrid component, auto-building missing time/structure score dependencies.")
    parser.add_argument("--dataset", type=str, default="Yelp-NOLA", choices=("Yelp-NOLA", "Yelp-PHL", "Yelp-TPA"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--time_dir", type=str, default="")
    parser.add_argument("--structure_dir", type=str, default="")
    parser.add_argument("--no_auto_run_dependencies", dest="auto_run_dependencies", action="store_false", default=True)

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

    parser.add_argument("--hybrid_train_topk", "--train_topk", dest="train_topk", type=int, default=120)
    parser.add_argument("--hybrid_val_early_stop_tailk", "--val_early_stop_tailk", dest="val_early_stop_tailk", type=int, default=5000)
    parser.add_argument("--hybrid_n_estimators", "--n_estimators", dest="n_estimators", type=int, default=1000)
    parser.add_argument("--hybrid_learning_rate", "--learning_rate", dest="learning_rate", type=float, default=0.03)
    parser.add_argument("--hybrid_num_leaves", "--num_leaves", dest="num_leaves", type=int, default=63)
    parser.add_argument("--hybrid_max_depth", "--max_depth", dest="max_depth", type=int, default=-1)
    parser.add_argument("--hybrid_min_child_samples", "--min_child_samples", dest="min_child_samples", type=int, default=50)
    parser.add_argument("--hybrid_reg_lambda", "--reg_lambda", dest="reg_lambda", type=float, default=1.0)
    parser.add_argument("--hybrid_reg_alpha", "--reg_alpha", dest="reg_alpha", type=float, default=0.0)
    parser.add_argument("--hybrid_subsample", "--subsample", dest="subsample", type=float, default=0.9)
    parser.add_argument("--hybrid_colsample_bytree", "--colsample_bytree", dest="colsample_bytree", type=float, default=0.9)
    parser.add_argument("--hybrid_early_stopping_rounds", "--early_stopping_rounds", dest="early_stopping_rounds", type=int, default=50)

    parser.add_argument("--force_time", action="store_true", default=False)
    parser.add_argument("--force_structure", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--force_feature_cache", action="store_true", default=False)
    parser.add_argument("--no_eval_test", dest="eval_test", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(load_args())
