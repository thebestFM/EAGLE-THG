import hashlib
import json
import math
import os
import os.path as osp
import time

import numpy as np
from scipy.sparse import load_npz

from utils import HIT_KS, add_metric_sums, finalize_metric_sums, ranking_metric_key


YELP_REL_NAMES = ["review_1star", "review_2star", "review_3star", "review_4star", "review_5star", "tip"]
BAD_CAT = 0
MID_CAT = 1
GOOD_CAT = 2
TIP_CAT = 3
CAT_NAMES = ["bad12", "mid3", "good45", "tip"]
SECONDS_PER_DAY = 86400.0
EPS = 1e-12


def relation_category(rel_id):
    rel_id = int(rel_id)
    if rel_id <= 1:
        return BAD_CAT
    if rel_id == 2:
        return MID_CAT
    if rel_id <= 4:
        return GOOD_CAT
    return TIP_CAT


def relation_kernel(query_rel, hist_rel):
    query_rel = int(query_rel)
    hist_rel = int(hist_rel)
    if query_rel == 5:
        if hist_rel == 5:
            return 1.0
        return 0.80 if hist_rel >= 3 else (0.45 if hist_rel == 2 else 0.15)
    if hist_rel == 5:
        return 0.85 if query_rel >= 3 else (0.45 if query_rel == 2 else 0.20)
    gap = abs(query_rel - hist_rel)
    if gap == 0:
        return 1.0
    if gap == 1:
        return 0.70
    if gap == 2:
        return 0.35
    return 0.08


REL_COMPAT = np.asarray(
    [[relation_kernel(q, h) for h in range(6)] for q in range(6)],
    dtype=np.float32,
)


def cat_counts(rel_counts):
    rel_counts = np.asarray(rel_counts, dtype=np.float32)
    return np.asarray(
        [
            rel_counts[0] + rel_counts[1],
            rel_counts[2],
            rel_counts[3] + rel_counts[4],
            rel_counts[5],
        ],
        dtype=np.float32,
    )


def star_average(rel_counts):
    rel_counts = np.asarray(rel_counts, dtype=np.float32)
    denom = float(np.sum(rel_counts[:5]))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(np.arange(1, 6, dtype=np.float32), rel_counts[:5]) / denom)


def age_weight(delta_seconds, half_life_days):
    half_life_days = float(half_life_days)
    if half_life_days <= 0.0:
        return 1.0
    return float(math.exp(-max(float(delta_seconds), 0.0) / (half_life_days * SECONDS_PER_DAY)))


def stable_hash(payload, length=12):
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[: int(length)]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prefix_metrics(prefix, metrics):
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items() if key != "profile"}


def add_metric_aliases(metrics):
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics


def format_metrics(metrics):
    return (
        f"mrr={metrics.get('mrr_strict', metrics.get('mrr', 0.0)):.5f} "
        f"h@1={metrics.get('hit@1_strict', metrics.get('hit1', 0.0)):.5f} "
        f"h@10={metrics.get('hit@10_strict', metrics.get('hit10', 0.0)):.5f}"
    )


def business_pool(data):
    return np.arange(int(data["business_first_id"]), int(data["business_last_id"]) + 1, dtype=np.int64)


def split_train_for_prediction(data):
    start = int(data.get("train_predict_start_idx", len(data["train_list"])))
    return data["train_list"][:start], data["train_list"][start:]


def snapshots_for_split(data, split):
    if split == "train":
        return data["train_list"][int(data.get("train_predict_start_idx", len(data["train_list"]))):]
    return data[f"{split}_list"]


def split_query_count(data, split):
    return int(sum(len(events) for events, _, _ in snapshots_for_split(data, split)))


def tail_query_window(data, split, tailk):
    total = split_query_count(data, split)
    half_start = total // 2
    tailk = int(tailk)
    start = half_start if tailk <= 0 else max(half_start, total - tailk)
    return int(start), int(total - start), int(total)


def make_valid_matrix(valid):
    return np.concatenate((np.ones((valid.shape[0], 1), dtype=bool), valid), axis=1)


def ranks_from_candidate_scores(scores, valid):
    pos = scores[:, :1]
    neg = scores[:, 1:]
    neg_valid = valid[:, 1:]
    loose = 1 + np.sum((neg > pos) & neg_valid, axis=1)
    strict = 1 + np.sum((neg >= pos) & neg_valid, axis=1)
    avg = (loose.astype(np.float64) + strict.astype(np.float64)) * 0.5
    return loose.astype(np.int64), strict.astype(np.int64), avg


def add_rank_sums(total, loose, strict, avg):
    total["count"] = total.get("count", 0) + int(len(loose))
    for name, ranks in (("loose", loose), ("strict", strict), ("avg", avg)):
        total[f"mrr_{name}"] = total.get(f"mrr_{name}", 0.0) + float(np.sum(1.0 / ranks))
        for k in HIT_KS:
            total[f"hit@{k}_{name}"] = total.get(f"hit@{k}_{name}", 0.0) + float(np.sum(ranks <= int(k)))


def finalize_rank_sums(sums):
    return add_metric_aliases(finalize_metric_sums(sums))


def dense_rank(scores, valid):
    masked = np.where(valid, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.int32)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, order.shape[1] + 1, dtype=np.int32)
    return np.where(valid, ranks, order.shape[1] + 1)


def zscore_by_query(scores, valid):
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(valid, scores, 0.0), axis=1, keepdims=True) / count
    var = np.sum(np.where(valid, (scores - mean) ** 2, 0.0), axis=1, keepdims=True) / count
    return np.where(valid, (scores - mean) / np.sqrt(np.maximum(var, EPS)), 0.0).astype(np.float32)


def minmax_by_query(scores, valid):
    low = np.min(np.where(valid, scores, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(valid, scores, -np.inf), axis=1, keepdims=True)
    denom = np.maximum(high - low, EPS)
    return np.where(valid, (scores - low) / denom, 0.0).astype(np.float32)


def ranker_matrix_cache_dir(root, dataset, seed, payload):
    h = stable_hash(payload, length=16)
    return ensure_dir(osp.join(root, str(dataset), f"seed{int(seed)}", f"m{h}"))


def ranker_matrix_cache_complete(cache_dir, name):
    return all(
        osp.exists(osp.join(cache_dir, f"{name}_{suffix}"))
        for suffix in ("X.npy", "y.npy", "group.npy", "info.json")
    )


def save_ranker_matrix_cache(cache_dir, name, X, y, group, info):
    ensure_dir(cache_dir)
    np.save(osp.join(cache_dir, f"{name}_X.npy"), X)
    np.save(osp.join(cache_dir, f"{name}_y.npy"), y)
    np.save(osp.join(cache_dir, f"{name}_group.npy"), group)
    save_json(osp.join(cache_dir, f"{name}_info.json"), info)


def load_ranker_matrix_cache(cache_dir, name):
    X = np.load(osp.join(cache_dir, f"{name}_X.npy"), mmap_mode=None)
    y = np.load(osp.join(cache_dir, f"{name}_y.npy"), mmap_mode=None)
    group = np.load(osp.join(cache_dir, f"{name}_group.npy"), mmap_mode=None)
    info = load_json(osp.join(cache_dir, f"{name}_info.json"))
    print(f"[THG-cache] loaded {name} ranker matrix <- {cache_dir}", flush=True)
    return X, y, group, info


class ScoreStore:
    def __init__(self, out_dir, mode):
        self.out_dir = out_dir
        self.mode = mode
        self.pos = np.load(osp.join(out_dir, f"{mode}_pos.npy"))
        if self.pos.ndim == 1:
            self.pos = self.pos.reshape(-1, 1)
        self.neg = load_npz(osp.join(out_dir, f"{mode}_neg.npz")).tocsr()
        self.valid_lens = np.load(osp.join(out_dir, f"{mode}_valid_lens.npy")).astype(np.int32)
        self.num_rows = int(self.pos.shape[0])
        self.max_negs = int(self.neg.shape[1])

    def get_block(self, start, end, width):
        width = min(int(width), self.max_negs)
        neg = self.neg[start:end, :width].toarray().astype(np.float32, copy=False)
        lens = np.minimum(self.valid_lens[start:end], width)
        mask = np.arange(width)[None, :] < lens[:, None]
        return (
            self.pos[start:end].astype(np.float32, copy=False),
            np.where(mask, neg, 0.0).astype(np.float32, copy=False),
            mask,
        )


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return float(time.time() - self.start)


def selection_score(metrics, metric):
    return float(metrics[ranking_metric_key(metric, strict=True)])
