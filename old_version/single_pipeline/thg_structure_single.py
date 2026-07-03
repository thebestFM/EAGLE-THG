import argparse
import gc
import json
import math
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from single_pipeline.thg_common import (
    REL_COMPAT,
    SECONDS_PER_DAY,
    add_rank_sums,
    business_pool,
    cat_counts,
    dense_rank,
    ensure_dir,
    finalize_rank_sums,
    format_metrics,
    make_valid_matrix,
    minmax_by_query,
    prefix_metrics,
    load_ranker_matrix_cache,
    ranker_matrix_cache_complete,
    ranker_matrix_cache_dir,
    ranks_from_candidate_scores,
    relation_category,
    selection_score,
    save_ranker_matrix_cache,
    snapshots_for_split,
    stable_hash,
    star_average,
    tail_query_window,
    zscore_by_query,
)
from utils import (
    ScoreWriter,
    collect_eval_batch,
    describe_loaded_data,
    is_run_complete,
    load_datasets,
    load_metrics,
    save_config,
    save_metrics,
    set_random_seed,
)


def current_rss_mb():
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except Exception:
        return 0.0


def haversine_block(lat_block, lon_block, lats, lons):
    radius = 6371.0088
    lat1 = np.radians(lat_block)[:, None]
    lon1 = np.radians(lon_block)[:, None]
    lat2 = np.radians(lats)[None, :]
    lon2 = np.radians(lons)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_geo_graph(data, topk, tau_km, min_weight=1e-4):
    ids = data["business_ids"].astype(np.int64, copy=False)
    lats = data["business_latitude"].astype(np.float64, copy=False)
    lons = data["business_longitude"].astype(np.float64, copy=False)
    n = len(ids)
    topk = min(max(0, int(topk)), max(n - 1, 0))
    neighbors = {int(bid): (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)) for bid in ids}
    if topk <= 0 or n <= 1:
        return neighbors
    tau_km = max(float(tau_km), 1e-6)
    try:
        from scipy.spatial import cKDTree

        lat0 = float(np.mean(lats))
        x = (lons - float(np.mean(lons))) * math.cos(math.radians(lat0)) * 111.320
        y = (lats - float(np.mean(lats))) * 110.574
        coords = np.stack((x, y), axis=1)
        tree = cKDTree(coords)
        dist, idx = tree.query(coords, k=topk + 1)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        for row in range(n):
            row_idx = idx[row, 1:]
            row_dist = dist[row, 1:]
            weight = np.exp(-row_dist / tau_km).astype(np.float32)
            keep = weight >= float(min_weight)
            neighbors[int(ids[row])] = (
                ids[row_idx[keep]].astype(np.int64, copy=False),
                weight[keep].astype(np.float32, copy=False),
            )
        return neighbors
    except Exception:
        pass

    block = 512
    for start in range(0, n, block):
        end = min(n, start + block)
        dist = haversine_block(lats[start:end], lons[start:end], lats, lons)
        dist[np.arange(end - start), np.arange(start, end)] = np.inf
        idx = np.argpartition(dist, topk, axis=1)[:, :topk]
        row_dist = np.take_along_axis(dist, idx, axis=1)
        order = np.argsort(row_dist, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        row_dist = np.take_along_axis(row_dist, order, axis=1)
        for local, global_idx in enumerate(range(start, end)):
            weight = np.exp(-row_dist[local] / tau_km).astype(np.float32)
            keep = weight >= float(min_weight)
            neighbors[int(ids[global_idx])] = (
                ids[idx[local][keep]].astype(np.int64, copy=False),
                weight[keep].astype(np.float32, copy=False),
            )
    return neighbors


def build_friend_graph(data):
    adj = defaultdict(set)
    for user, friend in data["static_user_friend_edges"].astype(np.int64, copy=False):
        if user == friend:
            continue
        adj[int(user)].add(int(friend))
        adj[int(friend)].add(int(user))
    out = {}
    for user, values in adj.items():
        arr = np.asarray(sorted(values), dtype=np.int64)
        out[int(user)] = arr
    return out


class BusinessStats:
    def __init__(self, data, half_life_days):
        self.first = int(data["business_first_id"])
        self.num_businesses = int(data["num_businesses"])
        self.counts = np.zeros((self.num_businesses, 6), dtype=np.float32)
        self.last_t = np.zeros(self.num_businesses, dtype=np.float64)
        self.seen = np.zeros(self.num_businesses, dtype=np.bool_)
        self.half_life_days = float(half_life_days)

    def _idx(self, business):
        return int(business) - self.first

    def _decay_indices(self, idx, raw_t):
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < self.num_businesses)]
        if len(idx) == 0 or self.half_life_days <= 0.0:
            if len(idx):
                self.seen[idx] = True
                self.last_t[idx] = np.maximum(self.last_t[idx], float(raw_t))
            return idx
        idx = np.unique(idx)
        active = self.seen[idx]
        if np.any(active):
            rows = idx[active]
            delta_days = np.maximum(0.0, (float(raw_t) - self.last_t[rows]) / SECONDS_PER_DAY)
            decay = np.exp(-delta_days / self.half_life_days).astype(np.float32)
            self.counts[rows] *= decay[:, None]
            self.last_t[rows] = float(raw_t)
        inactive = idx[~active]
        if len(inactive):
            self.seen[inactive] = True
            self.last_t[inactive] = float(raw_t)
        return idx

    def update(self, events, raw_t):
        if len(events) == 0:
            return
        idx = events[:, 2].astype(np.int64) - self.first
        self._decay_indices(idx, raw_t)
        for _, rel, business in events.astype(np.int64, copy=False):
            bi = self._idx(business)
            if 0 <= bi < self.num_businesses:
                self.counts[bi, int(rel)] += 1.0

    def get_counts(self, businesses, raw_t):
        businesses = np.asarray(businesses, dtype=np.int64).reshape(-1)
        idx = businesses - self.first
        self._decay_indices(idx, raw_t)
        out = np.zeros((len(businesses), 6), dtype=np.float32)
        valid = (idx >= 0) & (idx < self.num_businesses)
        out[valid] = self.counts[idx[valid]]
        return out


class UserPPRState:
    def __init__(self, data, geo_graph, half_life_days, direct_weight, geo_weight, min_value=1e-6):
        self.num_users = int(data["num_users"])
        self.geo_graph = geo_graph
        self.half_life_days = float(half_life_days)
        self.direct_weight = float(direct_weight)
        self.geo_weight = float(geo_weight)
        self.min_value = float(min_value)
        self.maps = [dict() for _ in range(self.num_users)]
        self.last_t = np.zeros(self.num_users, dtype=np.float64)
        self.seen = np.zeros(self.num_users, dtype=np.bool_)
        self.scale = np.ones(self.num_users, dtype=np.float64)

    def decay_user(self, user, raw_t):
        user = int(user)
        if user < 0 or user >= self.num_users:
            return
        if not self.seen[user]:
            self.seen[user] = True
            self.last_t[user] = float(raw_t)
            return
        if self.half_life_days <= 0.0:
            self.last_t[user] = float(raw_t)
            return
        delta_days = max(0.0, (float(raw_t) - self.last_t[user]) / SECONDS_PER_DAY)
        if delta_days <= 0.0:
            return
        self.scale[user] *= math.exp(-delta_days / self.half_life_days)
        if self.scale[user] < 1e-6:
            bucket = self.maps[user]
            for key in list(bucket.keys()):
                bucket[key] *= self.scale[user]
                if float(np.sum(bucket[key])) < self.min_value:
                    del bucket[key]
            self.scale[user] = 1.0
        self.last_t[user] = float(raw_t)

    def add(self, user, business, rel, value, raw_t):
        user = int(user)
        if user < 0 or user >= self.num_users:
            return
        self.decay_user(user, raw_t)
        vec = self.maps[user].get(int(business))
        if vec is None:
            vec = np.zeros(6, dtype=np.float32)
            self.maps[user][int(business)] = vec
        scale = max(float(self.scale[user]), 1e-12)
        vec[int(rel)] += float(value) / scale

    def update(self, events, raw_t):
        for user, rel, business in events.astype(np.int64, copy=False):
            self.add(user, business, rel, self.direct_weight, raw_t)
            neigh, weights = self.geo_graph.get(int(business), ((), ()))
            for nb, w in zip(neigh, weights):
                self.add(user, int(nb), rel, self.geo_weight * float(w), raw_t)

    def get_vec(self, user, business, raw_t):
        user = int(user)
        if user < 0 or user >= self.num_users:
            return np.zeros(6, dtype=np.float32)
        self.decay_user(user, raw_t)
        vec = self.maps[user].get(int(business))
        if vec is None:
            return np.zeros(6, dtype=np.float32)
        return (vec * float(self.scale[user])).astype(np.float32, copy=False)


class THGPPRContext:
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.geo_graph = build_geo_graph(data, args.geo_topk, args.geo_tau_km)
        self.friend_graph = build_friend_graph(data)
        self.business = BusinessStats(data, args.business_half_life_days)
        self.user = UserPPRState(
            data,
            self.geo_graph,
            args.user_half_life_days,
            args.direct_weight,
            args.geo_dynamic_weight,
        )

    def update(self, events, raw_t):
        self.business.update(events, raw_t)
        self.user.update(events, raw_t)

    def warmup(self, snapshots):
        for events, _, raw_t in snapshots:
            self.update(events, raw_t)
        return self


class StructureFeatureBuilder:
    def __init__(self):
        self.feature_names = [
            "ppr_score",
            "ppr_total_log1p",
            "ppr_same_cat",
            "ppr_good",
            "ppr_bad",
            "ppr_tip",
            "friend_score",
            "friend_total_log1p",
            "friend_same_cat",
            "friend_good",
            "friend_bad",
            "friend_tip",
            "business_total_log1p",
            "business_same_cat_rate",
            "business_good_rate",
            "business_bad_rate",
            "business_tip_rate",
            "business_avg_star",
            "business_star_gap",
            "geo_business_total_log1p",
            "geo_business_same_cat_rate",
            "geo_business_good_rate",
            "geo_business_bad_rate",
            "geo_business_tip_rate",
            "candidate_is_seen_by_user",
            "relation_id",
            "candidate_local_id",
            "base_score",
        ]

    def _rates(self, rel_counts):
        cats = cat_counts(rel_counts)
        denom = float(np.sum(cats))
        if denom <= 0.0:
            return cats, np.zeros(4, dtype=np.float32)
        return cats, (cats / denom).astype(np.float32)

    def _geo_business_counts(self, context, business, raw_t):
        neigh, weights = context.geo_graph.get(int(business), ((), ()))
        if len(neigh) == 0:
            return np.zeros(6, dtype=np.float32)
        counts = context.business.get_counts(np.asarray(neigh, dtype=np.int64), raw_t)
        return (counts * np.asarray(weights, dtype=np.float32).reshape(-1, 1)).sum(axis=0)

    def make(self, context, batch_data, cand_ids, valid, raw_t):
        data = context.data
        bsz, width = cand_ids.shape
        out = np.zeros((bsz, width, len(self.feature_names)), dtype=np.float32)
        flat_business = cand_ids.reshape(-1)
        business_counts = context.business.get_counts(flat_business, raw_t).reshape(bsz, width, 6)
        compat = REL_COMPAT
        for i in range(bsz):
            user = int(batch_data[i, 0])
            rel = int(batch_data[i, 1])
            rel_cat = relation_category(rel)
            friends = context.friend_graph.get(user, np.empty(0, dtype=np.int64))
            for j in range(width):
                if not valid[i, j]:
                    continue
                business = int(cand_ids[i, j])
                ppr_vec = context.user.get_vec(user, business, raw_t)
                ppr_cat, _ = self._rates(ppr_vec)
                ppr_score = float(np.dot(compat[rel], ppr_vec))
                friend_vec = np.zeros(6, dtype=np.float32)
                if len(friends):
                    for friend in friends:
                        friend_vec += context.user.get_vec(int(friend), business, raw_t)
                    friend_vec /= max(len(friends), 1)
                friend_cat, _ = self._rates(friend_vec)
                friend_score = float(np.dot(compat[rel], friend_vec))

                b_counts = business_counts[i, j]
                b_cat, b_rate = self._rates(b_counts)
                avg_star = star_average(b_counts)
                star_gap = abs((rel + 1) - avg_star) if rel <= 4 and avg_star > 0.0 else 0.0

                geo_counts = self._geo_business_counts(context, business, raw_t)
                geo_cat, geo_rate = self._rates(geo_counts)
                local_id = business - int(data["business_first_id"])
                base_score = (
                    ppr_score
                    + 0.75 * friend_score
                    + 0.60 * float(b_rate[rel_cat])
                    + 0.15 * math.log1p(float(np.sum(b_counts)))
                )
                values = [
                    ppr_score,
                    math.log1p(float(np.sum(ppr_vec))),
                    ppr_cat[rel_cat],
                    ppr_cat[2],
                    ppr_cat[0],
                    ppr_cat[3],
                    friend_score,
                    math.log1p(float(np.sum(friend_vec))),
                    friend_cat[rel_cat],
                    friend_cat[2],
                    friend_cat[0],
                    friend_cat[3],
                    math.log1p(float(np.sum(b_counts))),
                    b_rate[rel_cat],
                    b_rate[2],
                    b_rate[0],
                    b_rate[3],
                    avg_star,
                    star_gap,
                    math.log1p(float(np.sum(geo_counts))),
                    geo_rate[rel_cat],
                    geo_rate[2],
                    geo_rate[0],
                    geo_rate[3],
                    float(np.sum(ppr_vec) > 0.0),
                    float(rel),
                    float(local_id),
                    base_score,
                ]
                out[i, j, :] = np.asarray(values, dtype=np.float32)
        return out


def init_context(data, split, args):
    context = THGPPRContext(data, args)
    if split == "train":
        context.warmup(data["train_list"][: data["train_predict_start_idx"]])
    elif split == "val":
        context.warmup(data["train_list"])
    elif split == "test":
        context.warmup(data["train_list"])
        context.warmup(data["val_list"])
    else:
        raise ValueError(f"unknown split: {split}")
    return context


def iter_feature_blocks(data, split, feature_builder, args):
    context = init_context(data, split, args)
    for events, _, raw_t in snapshots_for_split(data, split):
        for batch, neg_arr, neg_mask in collect_eval_batch(events, raw_t, data["negative_sampler"], split, args.block_size):
            width = neg_arr.shape[1]
            valid = make_valid_matrix(neg_mask[:, :width])
            cand_ids = np.concatenate((batch[:, 2:3], neg_arr[:, :width]), axis=1)
            cube = feature_builder.make(context, batch, cand_ids, valid, raw_t)
            base_scores = cube[:, :, feature_builder.feature_names.index("base_score")]
            base_ranks = dense_rank(base_scores, valid)
            yield batch, valid, base_scores, base_ranks, cube
        context.update(events, raw_t)


def selected_for_training(base_ranks, valid, topk):
    selected = (base_ranks <= int(topk) + 1) & valid
    selected[:, 0] = True
    return selected


def build_ranker_matrix(data, split, feature_builder, args, topk=None, start_query=0, max_queries=None):
    X_parts, y_parts, groups = [], [], []
    queries = 0
    rows = 0
    seen_queries = 0
    done = False
    desc = f"thg_struct_build_{split}"
    if start_query or max_queries is not None:
        desc += "_slice"
    for _, valid, _, base_ranks, cube in tqdm(iter_feature_blocks(data, split, feature_builder, args), desc=desc, leave=False):
        selected = valid.copy() if topk is None else selected_for_training(base_ranks, valid, topk)
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
            X_parts.append(cube[i, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)
        if done:
            break
    if not X_parts:
        raise ValueError(f"no {split} rows for THG structure ranker")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {
            "queries": int(queries),
            "rows": int(rows),
            "start_query": int(start_query),
            "max_queries": None if max_queries is None else int(max_queries),
            "seen_queries": int(seen_queries),
        },
    )


def structure_matrix_cache_dir(args, split, topk, start_query, max_queries):
    payload = {
        "format": "thg_structure_ranker_matrix_v1",
        "dataset": args.dataset,
        "seed": args.seed,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "split": split,
        "topk": topk,
        "start_query": int(start_query),
        "max_queries": None if max_queries is None else int(max_queries),
        "geo_topk": args.geo_topk,
        "geo_tau_km": args.geo_tau_km,
        "friend_policy": "all",
        "user_half_life_days": args.user_half_life_days,
        "business_half_life_days": args.business_half_life_days,
        "direct_weight": args.direct_weight,
        "geo_dynamic_weight": args.geo_dynamic_weight,
    }
    return ranker_matrix_cache_dir("cache_thg_structure_ranker", args.dataset, args.seed, payload)


def build_or_load_ranker_matrix(data, split, feature_builder, args, topk=None, start_query=0, max_queries=None, cache_name=None):
    cache_name = cache_name or split
    cache_dir = structure_matrix_cache_dir(args, split, topk, start_query, max_queries)
    if ranker_matrix_cache_complete(cache_dir, cache_name) and not getattr(args, "force_feature_cache", False):
        return load_ranker_matrix_cache(cache_dir, cache_name)
    X, y, group, info = build_ranker_matrix(data, split, feature_builder, args, topk, start_query, max_queries)
    info = {**info, "cache_dir": cache_dir, "cache_name": cache_name}
    save_ranker_matrix_cache(cache_dir, cache_name, X, y, group, info)
    print(f"[THG-Structure] saved {cache_name} ranker matrix -> {cache_dir}", flush=True)
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
        raise RuntimeError("THG structure requires lightgbm") from exc
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


def predict_model(model, X):
    return model.predict(X).astype(np.float32, copy=False)


def evaluate_model(data, split, feature_builder, model, args, out_dir=None, write_scores=False):
    writer = ScoreWriter(out_dir, split) if write_scores else None
    sums = {}
    score_time = 0.0
    start = time.time()
    for _, valid, _, _, cube in tqdm(iter_feature_blocks(data, split, feature_builder, args), desc=f"thg_struct_eval_{split}", leave=False):
        scores = np.zeros(valid.shape, dtype=np.float32)
        flat_valid = valid.reshape(-1)
        if np.any(flat_valid):
            flat_cube = cube.reshape(-1, cube.shape[-1])
            t0 = time.time()
            scores.reshape(-1)[flat_valid] = predict_model(model, flat_cube[flat_valid])
            score_time += time.time() - t0
        loose, strict, avg = ranks_from_candidate_scores(scores, valid)
        add_rank_sums(sums, loose, strict, avg)
        if writer is not None:
            writer.write_batch(scores[:, :1], scores[:, 1:], valid[:, 1:])
    if writer is not None:
        writer.close()
    metrics = finalize_rank_sums(sums)
    metrics["profile"] = {"eval_time_sec": float(time.time() - start), "score_time_sec": float(score_time)}
    print(f"[THG-Structure] {split} {format_metrics(metrics)} time={metrics['profile']['eval_time_sec']:.1f}s", flush=True)
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
        "geo_topk": args.geo_topk,
        "geo_tau_km": args.geo_tau_km,
        "friend_policy": "all",
        "user_half_life_days": args.user_half_life_days,
        "business_half_life_days": args.business_half_life_days,
        "direct_weight": args.direct_weight,
        "geo_dynamic_weight": args.geo_dynamic_weight,
        "train_topk": args.train_topk,
        "val_early_stop_tailk": getattr(args, "val_early_stop_tailk", 8000),
        "lgbm": default_lgbm_params(args),
    }
    h = stable_hash(payload, length=12)
    name = (
        f"r{h}_geo{args.geo_topk}_frall"
        f"_traink{args.train_topk}_nsq{args.ns_q}_tpr{args.train_predict_ratio:g}"
    )
    return osp.join("results_thg_structure", args.dataset, f"seed{args.seed}", name)


def validate_args(args):
    if not hasattr(args, "val_early_stop_tailk"):
        args.val_early_stop_tailk = 8000
    if not hasattr(args, "force_feature_cache"):
        args.force_feature_cache = False
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or positive")
    if not 0.0 < float(args.train_predict_ratio) <= 1.0:
        raise ValueError("--train_predict_ratio must be in (0,1]")
    if int(args.train_topk) <= 0 or int(args.block_size) <= 0:
        raise ValueError("--train_topk and --block_size must be positive")
    if int(args.val_early_stop_tailk) < 0:
        raise ValueError("--val_early_stop_tailk must be non-negative")


def main(args):
    validate_args(args)
    set_random_seed(args.seed)
    out_dir = get_out_dir(args)
    expected_modes = ("train", "val")
    if getattr(args, "eval_test", True):
        expected_modes = expected_modes + ("test",)
    if is_run_complete(out_dir, expected_modes) and osp.isfile(osp.join(out_dir, "best_lgbm.txt")) and not getattr(args, "force", False):
        print(f"[THG-Structure] already complete: {out_dir}", flush=True)
        return load_metrics(out_dir)

    data = load_datasets(args.dataset, q=args.ns_q, load_train_ratio=args.train_predict_ratio, ns_seed=args.ns_seed)
    if not data.get("is_thg"):
        raise ValueError("THG structure requires a Yelp THG dataset")
    if not data["train_predict_count"]:
        raise ValueError("train_predict_ratio selected no train suffix for Structure-LGBM")
    describe_loaded_data(data, prefix="[THG-Structure]")
    ensure_dir(out_dir)
    start = time.time()
    feature_builder = StructureFeatureBuilder()
    X_train, y_train, group, train_info = build_or_load_ranker_matrix(data, "train", feature_builder, args, topk=args.train_topk)
    val_start, val_count, val_total = tail_query_window(data, "val", args.val_early_stop_tailk)
    X_val, y_val, group_val, val_info = build_or_load_ranker_matrix(
        data,
        "val",
        feature_builder,
        args,
        topk=None,
        start_query=val_start,
        max_queries=val_count,
        cache_name="val_es",
    )
    print(
        f"[THG-Structure] train rows={train_info['rows']} queries={train_info['queries']} "
        f"val_es rows={val_info['rows']} queries={val_info['queries']}/{val_total} "
        f"start={val_start} features={X_train.shape[1]}",
        flush=True,
    )
    model = fit_lgbm(X_train, y_train, group, X_val, y_val, group_val, feature_builder, args)
    del X_train, y_train, group, X_val, y_val, group_val
    gc.collect()

    train_metrics = evaluate_model(data, "train", feature_builder, model, args, out_dir, write_scores=True)
    val_metrics = evaluate_model(data, "val", feature_builder, model, args, out_dir, write_scores=True)
    test_metrics = None
    if getattr(args, "eval_test", True):
        test_metrics = evaluate_model(data, "test", feature_builder, model, args, out_dir, write_scores=True)
    model_path = osp.join(out_dir, "best_lgbm.txt")
    save_lgbm_model(model, model_path)
    metrics = {
        "format": "thg_structure_lgbm_v1",
        "model_path": model_path,
        "feature_names": feature_builder.feature_names,
        "train_info": train_info,
        "val_early_stop_info": val_info,
        "runtime_sec": float(time.time() - start),
        "rss_mb": current_rss_mb(),
    }
    metrics.update(prefix_metrics("train", train_metrics))
    metrics.update(prefix_metrics("val", val_metrics))
    if test_metrics is not None:
        metrics.update(prefix_metrics("test", test_metrics))
    save_config(out_dir, {**vars(args), "out_dir": out_dir, "friend_policy": "all", "feature_names": feature_builder.feature_names})
    save_metrics(out_dir, metrics)
    print(f"[THG-Structure] saved -> {out_dir}", flush=True)
    return metrics


def load_args():
    parser = argparse.ArgumentParser("Run the THG structure component.")
    parser.add_argument("--dataset", type=str, default="Yelp-NOLA", choices=("Yelp-NOLA", "Yelp-PHL", "Yelp-TPA"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--train_topk", type=int, default=100)
    parser.add_argument("--geo_topk", type=int, default=30)
    parser.add_argument("--geo_tau_km", type=float, default=1.0)
    parser.add_argument("--user_half_life_days", type=float, default=365.0)
    parser.add_argument("--business_half_life_days", type=float, default=365.0)
    parser.add_argument("--direct_weight", type=float, default=1.0)
    parser.add_argument("--geo_dynamic_weight", type=float, default=0.25)
    parser.add_argument("--structure_val_early_stop_tailk", "--val_early_stop_tailk", dest="val_early_stop_tailk", type=int, default=8000)
    parser.add_argument("--n_estimators", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=0.04)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--min_child_samples", type=int, default=50)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--force_feature_cache", action="store_true", default=False)
    parser.add_argument("--no_eval_test", dest="eval_test", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(load_args())
