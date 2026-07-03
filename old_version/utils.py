import json
import hashlib
import csv
import os
import os.path as osp
import pickle
import random
from types import SimpleNamespace

import numpy as np


TGB_DATASETS = ["tkgl-smallpedia", "tkgl-polecat", "tkgl-icews", "tkgl-wikidata"]
THG_DATASETS = {
    "Yelp-NOLA": "New_Orleans_LA",
    "Yelp-PHL": "Philadelphia_PA",
    "Yelp-TPA": "Tampa_FL",
}
NOT_TGB_DATASETS = [
    "ICEWS14",
    "ICEWS14s",
    "ICEWS18",
    "ICEWS05-15",
    "GDELT",
    "WIKI",
    "YAGO",
]
SUPPORTED_DATASETS = TGB_DATASETS + NOT_TGB_DATASETS + list(THG_DATASETS)
HIT_KS = [1, 3, 5, 10, 20, 50, 100]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    print(f"INFO: fixed random seed: {seed}", flush=True)


def inverse_aug(events, num_rels_raw, num_rels):
    inverse = events.copy()
    inverse[:, 0] = events[:, 2]
    inverse[:, 2] = events[:, 0]
    inverse[:, 1] = (events[:, 1] + int(num_rels_raw)) % int(num_rels)
    return np.vstack((events, inverse))


def _reformat_ts(timestamps, dataset_name='tkgl'):
    unique = np.array(sorted(np.unique(timestamps)), dtype=np.float64)
    if len(unique) <= 1:
        return np.zeros_like(timestamps, dtype=np.int64)

    diffs = np.diff(unique)
    step = float(np.min(diffs))
    if dataset_name in TGB_DATASETS and float(np.mean(diffs)) != step:
        if "icews" in dataset_name or "polecat" in dataset_name:
            step = 86400.0
        elif "wiki" in dataset_name or "yago" in dataset_name:
            step = 31536000.0
        else:
            step = float(np.mean(diffs))

    return np.ceil((timestamps - unique[0]) / step).astype(np.int64)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(path, obj):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _sparse_tools():
    from scipy.sparse import csr_matrix, save_npz, load_npz, vstack as sparse_vstack

    return csr_matrix, save_npz, load_npz, sparse_vstack


def _split_by_time(triples, ts_norm, ts_orig):
    snapshots = []
    for ts in sorted(np.unique(ts_norm)):
        mask = ts_norm == ts
        snapshots.append(
            (
                np.ascontiguousarray(triples[mask], dtype=np.int64),
                int(ts),
                int(ts_orig[mask][0]),
            )
        )
    return snapshots


def _split_events4_by_time(events4, time_to_norm=None):
    if len(events4) == 0:
        return []
    if len(events4) > 1 and np.any(events4[1:, 3] < events4[:-1, 3]):
        events4 = events4[np.argsort(events4[:, 3], kind="stable")]
    snapshots = []
    raw_times = events4[:, 3].astype(np.int64, copy=False)
    start = 0
    while start < len(events4):
        raw_ts = int(raw_times[start])
        end = start + 1
        while end < len(events4) and int(raw_times[end]) == raw_ts:
            end += 1
        t_norm = int(time_to_norm[int(raw_ts)] if time_to_norm is not None else raw_ts)
        snapshots.append(
            (
                np.ascontiguousarray(events4[start:end, :3], dtype=np.int64),
                t_norm,
                int(raw_ts),
            )
        )
        start = end
    return snapshots


def _selected_train_suffix(train_list, ratio):
    if ratio <= 0:
        return len(train_list), []
    count = int(len(train_list) * float(ratio))
    start = len(train_list) - count
    return start, train_list[start:]


def _events4_from_snapshots(snapshot_list):
    chunks = []
    for events, _, t_orig in snapshot_list:
        if len(events) == 0:
            continue
        t_col = np.full((len(events), 1), int(t_orig), dtype=np.int64)
        chunks.append(np.hstack((events.astype(np.int64, copy=False), t_col)))
    if not chunks:
        return np.empty((0, 4), dtype=np.int64)
    return np.vstack(chunks)


def _positive_maps(events4):
    by_query = {}
    by_time_rel = {}
    for s, r, o, t in events4.astype(np.int64, copy=False):
        by_query.setdefault((int(t), int(s), int(r)), set()).add(int(o))
        by_time_rel.setdefault((int(t), int(r)), set()).add(int(o))
    by_query = {k: np.array(sorted(v), dtype=np.int64) for k, v in by_query.items()}
    by_time_rel = {k: np.array(sorted(v), dtype=np.int64) for k, v in by_time_rel.items()}
    return by_query, by_time_rel


def _cache_path(root, dataset_name, mode, q, seed, load_train_ratio):
    ratio = f"{float(load_train_ratio):.8g}"
    name = f"{dataset_name}_{mode}_nsq{int(q)}_seed{int(seed)}_trainratio{ratio}_same_t_rel_v2.pkl"
    return osp.join(root, name)


def _thg_cache_path(root, dataset_name, mode, q, seed, load_train_ratio):
    ratio = f"{float(load_train_ratio):.8g}"
    name = f"{dataset_name}_{mode}_nsq{int(q)}_seed{int(seed)}_trainratio{ratio}_business_reject_v2.pkl"
    return osp.join(root, name)


def _load_or_create_full_negatives(snapshot_list, cache_path):
    if osp.exists(cache_path):
        print(f"[data] loading full negatives: {cache_path}", flush=True)
        payload = _load_pkl(cache_path)
        if isinstance(payload, dict) and payload.get("__format__") == "thg_full_negatives_v1":
            return payload["query_pos"]
        return payload

    query_pos, _ = _positive_maps(_events4_from_snapshots(snapshot_list))
    _save_pkl(
        cache_path,
        {
            "__format__": "thg_full_negatives_v1",
            "description": "Compact cache for ns_q=-1: negatives are all business ids except positives for the same (ts, head, rel).",
            "query_pos": query_pos,
        },
    )
    print(f"[data] saved full negatives: {cache_path}", flush=True)
    return query_pos


def _available_negative_stats(snapshot_list, all_dst):
    query_pos, _ = _positive_maps(_events4_from_snapshots(snapshot_list))
    total_dst = int(len(all_dst))
    available = []
    for positives in query_pos.values():
        positive_count = int(len(positives))
        available.append(max(total_dst - positive_count, 0))
    if not available:
        return query_pos, {"queries": 0, "min_available": total_dst, "available": np.empty(0, dtype=np.int64)}
    arr = np.asarray(available, dtype=np.int64)
    return query_pos, {"queries": int(len(arr)), "min_available": int(arr.min()), "available": arr}


def _sample_without_replace(rng, pool, count):
    if count <= 0 or len(pool) == 0:
        return np.empty(0, dtype=np.int64)
    if len(pool) <= count:
        return np.asarray(pool, dtype=np.int64).copy()
    return rng.choice(pool, count, replace=False)


def _sample_negatives(snapshot_list, all_dst, q, seed, cache_path):
    if osp.exists(cache_path):
        print(f"[data] loading negatives: {cache_path}", flush=True)
        return _load_pkl(cache_path)

    rng = np.random.RandomState(seed)
    query_pos, time_rel_pos = _positive_maps(_events4_from_snapshots(snapshot_list))
    sampled = {}
    for key, positives in query_pos.items():
        t, _, r = key
        same_time_rel = time_rel_pos.get((t, r), positives)

        primary_pool = np.setdiff1d(all_dst, same_time_rel, assume_unique=False)
        parts = [_sample_without_replace(rng, primary_pool, int(q))]
        current = sum(len(part) for part in parts)

        if current < int(q):
            fill_pool = np.setdiff1d(same_time_rel, positives, assume_unique=False)
            parts.append(_sample_without_replace(rng, fill_pool, int(q) - current))
            current = sum(len(part) for part in parts)

        if current < int(q):
            fallback_pool = np.setdiff1d(all_dst, positives, assume_unique=False)
            need = int(q) - current
            if len(fallback_pool) == 0:
                fallback_pool = all_dst
            parts.append(rng.choice(fallback_pool, need, replace=len(fallback_pool) < need))

        sampled[key] = np.concatenate(parts)[: int(q)].astype(np.int32, copy=False)

    _save_pkl(cache_path, sampled)
    print(f"[data] saved negatives: {cache_path}", flush=True)
    return sampled


def _sample_negatives_rejection(snapshot_list, all_dst, q, seed, cache_path):
    if osp.exists(cache_path):
        print(f"[data] loading negatives: {cache_path}", flush=True)
        return _load_pkl(cache_path)

    rng = np.random.RandomState(seed)
    query_pos, time_rel_pos = _positive_maps(_events4_from_snapshots(snapshot_list))
    all_dst = np.asarray(all_dst, dtype=np.int64)
    sampled = {}
    for key, positives in query_pos.items():
        t, _, r = key
        positives_set = set(int(x) for x in positives)
        same_time_rel = time_rel_pos.get((t, r), positives)
        primary_exclude = set(int(x) for x in same_time_rel)
        selected = []
        seen = set()
        attempts = 0
        while len(selected) < int(q) and attempts < max(64, int(q) * 8):
            need = int(q) - len(selected)
            draw = all_dst[rng.randint(0, len(all_dst), size=max(need * 2, 16))]
            for value in draw:
                value = int(value)
                if value in primary_exclude or value in seen:
                    continue
                selected.append(value)
                seen.add(value)
                if len(selected) == int(q):
                    break
            attempts += len(draw)
        if len(selected) < int(q):
            for value in all_dst:
                value = int(value)
                if value in primary_exclude or value in seen:
                    continue
                selected.append(value)
                seen.add(value)
                if len(selected) == int(q):
                    break
        if len(selected) < int(q):
            fill = [int(x) for x in same_time_rel if int(x) not in positives_set and int(x) not in seen]
            if fill:
                take = min(int(q) - len(selected), len(fill))
                extra = _sample_without_replace(rng, np.asarray(fill, dtype=np.int64), take)
                selected.extend(int(x) for x in extra)
                seen.update(int(x) for x in extra)
        if len(selected) < int(q):
            fallback = [int(x) for x in all_dst if int(x) not in positives_set and int(x) not in seen]
            if fallback:
                take = min(int(q) - len(selected), len(fallback))
                extra = _sample_without_replace(rng, np.asarray(fallback, dtype=np.int64), take)
                selected.extend(int(x) for x in extra)
        sampled[key] = np.asarray(selected[: int(q)], dtype=np.int32)

    _save_pkl(cache_path, sampled)
    print(f"[data] saved negatives: {cache_path}", flush=True)
    return sampled


class NegativeSampler:
    def __init__(self, num_nodes, first_dst_id=0, last_dst_id=None, base_sampler=None):
        self.base_sampler = base_sampler
        self.eval_set = {}
        self.full_modes = {}
        self.strategy = "dst-time-filtered"
        self.first_dst_id = int(first_dst_id)
        self.last_dst_id = int(num_nodes - 1 if last_dst_id is None else last_dst_id)
        self.all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1, dtype=np.int64)

    def add_full_mode(self, mode, snapshot_list=None, positives=None):
        if positives is None:
            positives = _positive_maps(_events4_from_snapshots(snapshot_list))[0]
        self.full_modes[mode] = positives

    def add_sampled_mode(self, mode, sampled):
        self.eval_set[mode] = sampled

    def query_batch(self, sources, destinations, timestamps, relations, mode):
        if mode in self.full_modes:
            positives = self.full_modes[mode]
            return [
                np.setdiff1d(
                    self.all_dst,
                    positives.get((int(t), int(s), int(r)), np.array([int(o)], dtype=np.int64)),
                    assume_unique=False,
                ).astype(np.int32, copy=False)
                for s, o, t, r in zip(sources, destinations, timestamps, relations)
            ]

        if mode in self.eval_set:
            data = self.eval_set[mode]
            empty = np.empty(0, dtype=np.int32)
            return [data.get((int(t), int(s), int(r)), empty) for s, t, r in zip(sources, timestamps, relations)]

        if self.base_sampler is not None:
            return self.base_sampler.query_batch(sources, destinations, timestamps, relations, mode)

        return [np.empty(0, dtype=np.int32) for _ in sources]


def _attach_negatives(
    sampler,
    root,
    dataset_name,
    mode,
    snapshot_list,
    q,
    seed,
    load_train_ratio,
):
    if not snapshot_list:
        return
    if q == -1:
        sampler.add_full_mode(mode, snapshot_list)
        return
    path = _cache_path(root, dataset_name, mode, q, seed, load_train_ratio)
    sampled = _sample_negatives(snapshot_list, sampler.all_dst, q, seed, path)
    sampler.add_sampled_mode(mode, sampled)


def _attach_thg_negatives(
    sampler,
    root,
    dataset_name,
    mode,
    snapshot_list,
    q,
    seed,
    load_train_ratio,
):
    if not snapshot_list:
        return
    full_path = _thg_cache_path(root, dataset_name, mode, -1, seed, load_train_ratio)
    if q == -1:
        positives = _load_or_create_full_negatives(snapshot_list, full_path)
        sampler.add_full_mode(mode, positives=positives)
        return
    _, stats = _available_negative_stats(snapshot_list, sampler.all_dst)
    if int(stats["min_available"]) < int(q):
        below = int(np.sum(stats["available"] < int(q)))
        print(
            f"[data] {dataset_name} {mode}: {below}/{stats['queries']} queries have fewer than "
            f"ns_q={int(q)} available business negatives (min_available={stats['min_available']}); "
            "falling back to ns_q=-1/full business negatives for this split.",
            flush=True,
        )
        positives = _load_or_create_full_negatives(snapshot_list, full_path)
        sampler.add_full_mode(mode, positives=positives)
        return
    path = _thg_cache_path(root, dataset_name, mode, q, seed, load_train_ratio)
    sampled = _sample_negatives_rejection(snapshot_list, sampler.all_dst, q, seed, path)
    sampler.add_sampled_mode(mode, sampled)


def load_datasets(dataset_name, q=-1, load_train_ratio=0.0, load_eval_neg=True, ns_seed=42):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    if q == 0 or q < -1:
        raise ValueError("q must be -1 or a positive integer")
    if not 0.0 <= float(load_train_ratio) <= 1.0:
        raise ValueError("load_train_ratio must be in [0, 1]")
    if dataset_name in THG_DATASETS:
        return load_dataset_thg(dataset_name, q, load_train_ratio, load_eval_neg, ns_seed)
    if dataset_name in TGB_DATASETS:
        return load_dataset_tgb(dataset_name, q, load_train_ratio, load_eval_neg, ns_seed)
    return load_dataset_tkg(dataset_name, q, load_train_ratio, load_eval_neg, ns_seed)


def load_dataset_tgb(dataset_name, q=-1, load_train_ratio=0.0, load_eval_neg=True, ns_seed=42):
    import sys
    custom_tgb_path = '/export/data/fteng/EAGLE-new/TGB2-TKGLP'
    if custom_tgb_path not in sys.path:
        sys.path.insert(0, custom_tgb_path)
    from tgb.linkproppred.dataset import LinkPropPredDataset

    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
    num_nodes = int(dataset.num_nodes)
    num_rels = int(dataset.num_rels)
    num_rels_raw = num_rels // 2

    ts_orig = dataset.full_data["timestamps"]
    ts_norm = _reformat_ts(ts_orig, dataset_name)
    triples = np.stack((dataset.full_data["sources"], dataset.edge_type, dataset.full_data["destinations"]), axis=1)

    train_list = _split_by_time(triples[dataset.train_mask], ts_norm[dataset.train_mask], ts_orig[dataset.train_mask])
    val_list = _split_by_time(triples[dataset.val_mask], ts_norm[dataset.val_mask], ts_orig[dataset.val_mask])
    test_list = _split_by_time(triples[dataset.test_mask], ts_norm[dataset.test_mask], ts_orig[dataset.test_mask])

    base_sampler = dataset.negative_sampler
    if load_eval_neg and q == -1:
        dataset.load_val_ns()
        dataset.load_test_ns()
        base_sampler = dataset.negative_sampler

    first = getattr(base_sampler, "first_dst_id", 0)
    last = getattr(base_sampler, "last_dst_id", num_nodes - 1)
    sampler = NegativeSampler(num_nodes, first, last, base_sampler=base_sampler if q == -1 else None)

    train_predict_start, train_predict_list = _selected_train_suffix(train_list, load_train_ratio)
    if train_predict_list:
        _attach_negatives(sampler, dataset.root, dataset_name, "train", train_predict_list, q, ns_seed, load_train_ratio)

    if load_eval_neg and q > 0:
        _attach_negatives(sampler, dataset.root, dataset_name, "val", val_list, q, ns_seed, load_train_ratio)
        _attach_negatives(sampler, dataset.root, dataset_name, "test", test_list, q, ns_seed, load_train_ratio)

    # TGB exposes dataset.negative_sampler as a read-only property in some versions.
    # Keep EAGLE's train/val/test sampler in the returned data dict instead.
    return {
        "dataset": dataset,
        "negative_sampler": sampler,
        "name": dataset_name,
        "is_tgb": True,
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "num_rels_raw": num_rels_raw,
        "train_list": train_list,
        "val_list": val_list,
        "test_list": test_list,
        "train_predict_start_idx": train_predict_start,
        "train_predict_count": len(train_predict_list),
        "timestamps_norm_max": int(np.max(ts_norm)),
    }


def _read_thg_events(dataset_dir, split):
    path = osp.join(dataset_dir, f"{split}_events.csv")
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(row["user_id"]),
                    int(row["rel_id"]),
                    int(row["business_id"]),
                    int(row["ts"]),
                )
            )
    if not rows:
        return np.empty((0, 4), dtype=np.int64)
    arr = np.asarray(rows, dtype=np.int64)
    if len(arr) > 1 and np.any(arr[1:, 3] < arr[:-1, 3]):
        arr = arr[np.argsort(arr[:, 3], kind="stable")]
    return np.ascontiguousarray(arr, dtype=np.int64)


def _read_thg_static_edges(dataset_dir):
    path = osp.join(dataset_dir, "static_user_friend_edges.csv")
    if not osp.isfile(path):
        return np.empty((0, 2), dtype=np.int64)
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["user_id"]), int(row["friend_user_id"])))
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _read_thg_business_geo(dataset_dir):
    path = osp.join(dataset_dir, "business_id_map.csv")
    ids, lats, lons = [], [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["business_id"]))
            lats.append(float(row["latitude"]))
            lons.append(float(row["longitude"]))
    if not ids:
        return {
            "business_ids": np.empty(0, dtype=np.int64),
            "latitude": np.empty(0, dtype=np.float32),
            "longitude": np.empty(0, dtype=np.float32),
        }
    order = np.argsort(np.asarray(ids, dtype=np.int64), kind="stable")
    return {
        "business_ids": np.asarray(ids, dtype=np.int64)[order],
        "latitude": np.asarray(lats, dtype=np.float32)[order],
        "longitude": np.asarray(lons, dtype=np.float32)[order],
    }


def _read_thg_stats(dataset_dir):
    with open(osp.join(dataset_dir, "model_input_stats.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_thg(dataset_name, q=-1, load_train_ratio=0.0, load_eval_neg=True, ns_seed=42):
    if dataset_name not in THG_DATASETS:
        raise ValueError(f"unknown THG dataset: {dataset_name}")

    data_root = osp.join(osp.dirname(osp.abspath(__file__)), "Yelp_datasets")
    dataset_dir = osp.join(data_root, THG_DATASETS[dataset_name])
    stats = _read_thg_stats(dataset_dir)
    num_users = int(stats["num_users"])
    num_businesses = int(stats["num_businesses"])
    num_nodes = int(stats["num_entities_user_business_disjoint"])
    num_rels = int(stats["num_relations"])
    business_first = int(stats["business_id_range"][0])
    business_last = int(stats["business_id_range"][1])

    train_raw = _read_thg_events(dataset_dir, "train")
    val_raw = _read_thg_events(dataset_dir, "valid")
    test_raw = _read_thg_events(dataset_dir, "test")
    all_times = np.array(
        sorted(np.unique(np.concatenate((train_raw[:, 3], val_raw[:, 3], test_raw[:, 3])))),
        dtype=np.int64,
    )
    time_to_norm = {int(t): i for i, t in enumerate(all_times)}

    train_list = _split_events4_by_time(train_raw, time_to_norm)
    val_list = _split_events4_by_time(val_raw, time_to_norm)
    test_list = _split_events4_by_time(test_raw, time_to_norm)

    sampler = NegativeSampler(num_nodes, first_dst_id=business_first, last_dst_id=business_last)
    train_predict_start, train_predict_list = _selected_train_suffix(train_list, load_train_ratio)
    if train_predict_list:
        _attach_thg_negatives(sampler, dataset_dir, dataset_name, "train", train_predict_list, q, ns_seed, load_train_ratio)
    if load_eval_neg:
        _attach_thg_negatives(sampler, dataset_dir, dataset_name, "val", val_list, q, ns_seed, load_train_ratio)
        _attach_thg_negatives(sampler, dataset_dir, dataset_name, "test", test_list, q, ns_seed, load_train_ratio)

    all_events = _events4_from_snapshots(train_list + val_list + test_list)
    static_friend_edges = _read_thg_static_edges(dataset_dir)
    business_geo = _read_thg_business_geo(dataset_dir)
    dataset = SimpleNamespace(
        name=dataset_name,
        root=dataset_dir,
        num_nodes=num_nodes,
        num_rels=num_rels,
        num_rels_raw=num_rels,
        num_users=num_users,
        num_businesses=num_businesses,
        business_first_id=business_first,
        business_last_id=business_last,
        negative_sampler=sampler,
        full_data={
            "sources": all_events[:, 0],
            "edge_type": all_events[:, 1],
            "destinations": all_events[:, 2],
            "timestamps": all_events[:, 3],
            "edge_idxs": np.arange(len(all_events), dtype=np.int64),
        },
    )

    return {
        "dataset": dataset,
        "negative_sampler": sampler,
        "name": dataset_name,
        "is_tgb": False,
        "is_thg": True,
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "num_rels_raw": num_rels,
        "num_users": num_users,
        "num_businesses": num_businesses,
        "business_first_id": business_first,
        "business_last_id": business_last,
        "business_ids": business_geo["business_ids"],
        "business_latitude": business_geo["latitude"],
        "business_longitude": business_geo["longitude"],
        "static_user_friend_edges": static_friend_edges,
        "relation2id": dict(stats.get("relation2id", {})),
        "train_list": train_list,
        "val_list": val_list,
        "test_list": test_list,
        "train_predict_start_idx": train_predict_start,
        "train_predict_count": len(train_predict_list),
        "timestamps_norm_max": int(len(all_times) - 1),
        "raw_timestamps": all_times,
        "raw_ts_to_cont": time_to_norm,
        "root": dataset_dir,
    }


def _read_tkg_stat(dataset_dir):
    with open(osp.join(dataset_dir, "stat.txt"), "r") as f:
        cols = f.readline().split()
    return int(cols[0]), int(cols[1])


def _read_tkg_split(dataset_dir, split):
    path = osp.join(dataset_dir, f"{split}.txt")
    arr = np.loadtxt(path, dtype=np.int64, usecols=(0, 1, 2, 3), ndmin=2)
    if len(arr) > 1 and np.any(arr[1:, 3] < arr[:-1, 3]):
        arr = arr[np.argsort(arr[:, 3], kind="stable")]
    return np.ascontiguousarray(arr, dtype=np.int64)


def _augment_tkg_snapshots(events4, time_to_norm, num_rels_raw):
    snapshots = []
    for t in sorted(np.unique(events4[:, 3])):
        chunk = events4[events4[:, 3] == t]
        forward = chunk[:, [0, 1, 2]]
        inverse = np.stack((chunk[:, 2], chunk[:, 1] + num_rels_raw, chunk[:, 0]), axis=1)
        snapshots.append((np.vstack((forward, inverse)).astype(np.int64, copy=False), int(time_to_norm[int(t)]), int(t)))
    return snapshots


def load_dataset_tkg(dataset_name, q=-1, load_train_ratio=0.0, load_eval_neg=True, ns_seed=42):
    if dataset_name not in NOT_TGB_DATASETS:
        raise ValueError(f"unknown non-TGB dataset: {dataset_name}")

    data_root = osp.join(osp.dirname(osp.abspath(__file__)), "data")
    dataset_dir = osp.join(data_root, dataset_name)
    num_nodes, num_rels_raw = _read_tkg_stat(dataset_dir)

    train_raw = _read_tkg_split(dataset_dir, "train")
    val_raw = _read_tkg_split(dataset_dir, "valid")
    test_raw = _read_tkg_split(dataset_dir, "test")

    all_times = np.array(sorted(np.unique(np.concatenate((train_raw[:, 3], val_raw[:, 3], test_raw[:, 3])))), dtype=np.int64)
    time_to_norm = {int(t): i for i, t in enumerate(all_times)}

    train_list = _augment_tkg_snapshots(train_raw, time_to_norm, num_rels_raw)
    val_list = _augment_tkg_snapshots(val_raw, time_to_norm, num_rels_raw)
    test_list = _augment_tkg_snapshots(test_raw, time_to_norm, num_rels_raw)

    sampler = NegativeSampler(num_nodes)
    train_predict_start, train_predict_list = _selected_train_suffix(train_list, load_train_ratio)
    if train_predict_list:
        _attach_negatives(sampler, dataset_dir, dataset_name, "train", train_predict_list, q, ns_seed, load_train_ratio)
    if load_eval_neg:
        _attach_negatives(sampler, dataset_dir, dataset_name, "val", val_list, q, ns_seed, load_train_ratio)
        _attach_negatives(sampler, dataset_dir, dataset_name, "test", test_list, q, ns_seed, load_train_ratio)

    all_aug = _events4_from_snapshots(train_list + val_list + test_list)
    dataset = SimpleNamespace(
        name=dataset_name,
        root=dataset_dir,
        num_nodes=int(num_nodes),
        num_rels=int(num_rels_raw * 2),
        num_rels_raw=int(num_rels_raw),
        edge_feat=None,
        node_feat=None,
        negative_sampler=sampler,
        full_data={
            "sources": all_aug[:, 0],
            "edge_type": all_aug[:, 1],
            "destinations": all_aug[:, 2],
            "timestamps": all_aug[:, 3],
            "edge_idxs": np.arange(len(all_aug), dtype=np.int64),
        },
    )

    return {
        "dataset": dataset,
        "negative_sampler": sampler,
        "name": dataset_name,
        "is_tgb": False,
        "num_nodes": int(num_nodes),
        "num_rels": int(num_rels_raw * 2),
        "num_rels_raw": int(num_rels_raw),
        "train_list": train_list,
        "val_list": val_list,
        "test_list": test_list,
        "train_predict_start_idx": train_predict_start,
        "train_predict_count": len(train_predict_list),
        "timestamps_norm_max": int(len(all_times) - 1),
    }


def collect_eval_batch(events, ts_orig, neg_sampler, mode, batch_size):
    for start in range(0, len(events), batch_size):
        batch = np.ascontiguousarray(events[start : start + batch_size], dtype=np.int64)
        ts = np.full(len(batch), int(ts_orig), dtype=np.int64)
        negs = neg_sampler.query_batch(batch[:, 0], batch[:, 2], ts, batch[:, 1], mode)
        width = max((len(x) for x in negs), default=0) or 1
        neg_arr = np.full((len(batch), width), -1, dtype=np.int64)
        for i, row in enumerate(negs):
            if len(row):
                neg_arr[i, : len(row)] = row
        yield batch, neg_arr, neg_arr != -1


def compute_ranks(pos_scores, neg_scores, neg_mask):
    pos = np.asarray(pos_scores, dtype=np.float32).reshape(-1, 1)
    neg = np.asarray(neg_scores, dtype=np.float32)
    mask = np.asarray(neg_mask, dtype=bool)
    loose = 1 + np.sum((neg > pos) & mask, axis=1)
    strict = 1 + np.sum((neg >= pos) & mask, axis=1)
    # Average tie rank: expected rank if ties are broken uniformly at random.
    avg = (loose.astype(np.float64) + strict.astype(np.float64)) * 0.5
    return loose.astype(np.int64), strict.astype(np.int64), avg


def compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask, hit_ks=HIT_KS):
    loose_rank, strict_rank, avg_rank = compute_ranks(pos_scores, neg_scores, neg_mask)
    sums = {"count": int(len(loose_rank))}
    for name, ranks in (("loose", loose_rank), ("strict", strict_rank), ("avg", avg_rank)):
        sums[f"mrr_{name}"] = float(np.sum(1.0 / ranks))
        for k in hit_ks:
            sums[f"hit@{k}_{name}"] = float(np.sum(ranks <= int(k)))
    return sums


def ranking_metric_key(metric, strict=True):
    suffix = "strict" if strict else "loose"
    text = str(metric).strip().lower().replace("_", "").replace("-", "").replace("@", "")
    if text in ("mrr", "strictmrr", "loosemrr"):
        return f"mrr_{suffix}"
    if text.startswith("hit"):
        k = text[3:]
    elif text.startswith("hr"):
        k = text[2:]
    else:
        raise ValueError(f"unsupported ranking metric: {metric}")
    if not k.isdigit():
        raise ValueError(f"unsupported ranking metric: {metric}")
    return f"hit@{int(k)}_{suffix}"


def get_ranking_metric(metrics, metric, split=None, strict=True):
    key = ranking_metric_key(metric, strict=strict)
    if split is not None:
        key = f"{split}_{key}"
    return float(metrics[key])


def add_metric_sums(total, batch):
    for key, value in batch.items():
        total[key] = total.get(key, 0.0) + value


def finalize_metric_sums(sums, hit_ks=HIT_KS):
    count = int(sums.get("count", 0))
    if count <= 0:
        metrics = {"mrr_loose": 0.0, "mrr_strict": 0.0, "mrr_avg": 0.0}
        for k in hit_ks:
            metrics[f"hit@{k}_loose"] = 0.0
            metrics[f"hit@{k}_strict"] = 0.0
            metrics[f"hit@{k}_avg"] = 0.0
        return metrics
    return {key: float(value) / count for key, value in sums.items() if key != "count"}


class ScoreWriter:
    def __init__(self, out_dir, mode):
        self.out_dir = out_dir
        self.mode = mode
        self.pos_chunks = []
        self.neg_chunks = []
        self.valid_lens = []
        self.max_negs = 0
        self.num_rows = 0
        os.makedirs(out_dir, exist_ok=True)

    def write_batch(self, pos_scores, neg_scores, neg_mask):
        csr_matrix, _, _, _ = _sparse_tools()
        pos = np.asarray(pos_scores, dtype=np.float32).reshape(-1, 1)
        neg = np.where(neg_mask, neg_scores, 0.0).astype(np.float32, copy=False)
        neg[np.abs(neg) < 1e-12] = 0.0
        self.pos_chunks.append(pos)
        self.neg_chunks.append(csr_matrix(neg))
        self.valid_lens.append(np.sum(neg_mask, axis=1).astype(np.int32))
        self.max_negs = max(self.max_negs, neg.shape[1])
        self.num_rows += pos.shape[0]

    def close(self):
        if self.num_rows == 0:
            return
        csr_matrix, save_npz, _, sparse_vstack = _sparse_tools()
        chunks = []
        for chunk in self.neg_chunks:
            if chunk.shape[1] == self.max_negs:
                chunks.append(chunk)
            else:
                chunks.append(csr_matrix((chunk.data, chunk.indices, chunk.indptr), shape=(chunk.shape[0], self.max_negs)))
        np.save(osp.join(self.out_dir, f"{self.mode}_pos.npy"), np.vstack(self.pos_chunks))
        np.save(osp.join(self.out_dir, f"{self.mode}_valid_lens.npy"), np.concatenate(self.valid_lens))
        save_npz(osp.join(self.out_dir, f"{self.mode}_neg.npz"), sparse_vstack(chunks, format="csr"), compressed=False)
        meta = {
            "format": "stream_sparse_v1",
            "num_rows": int(self.num_rows),
            "max_negs": int(self.max_negs),
            "neg_default_kind": "zero",
            "neg_default_scalar": 0.0,
        }
        with open(osp.join(self.out_dir, f"{self.mode}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def describe_loaded_data(data, prefix="[data]"):
    def stats(name):
        snaps = data[name]
        return len(snaps), sum(len(events) for events, _, _ in snaps)

    train = stats("train_list")
    val = stats("val_list")
    test = stats("test_list")
    print(
        f"{prefix} {data['name']}: train={train[0]} ts/{train[1]} events, "
        f"val={val[0]} ts/{val[1]} events, test={test[0]} ts/{test[1]} events, "
        f"nodes={data['num_nodes']}, rels={data['num_rels']} raw_rels={data['num_rels_raw']}",
        flush=True,
    )


def get_negative_sampler(data):
    if "negative_sampler" in data:
        return data["negative_sampler"]
    return data["dataset"].negative_sampler


def get_destination_pool(data, num_nodes):
    sampler = get_negative_sampler(data)
    if hasattr(sampler, "first_dst_id") and hasattr(sampler, "last_dst_id"):
        first_dst = int(sampler.first_dst_id)
        last_dst = int(sampler.last_dst_id)
        if first_dst <= last_dst:
            return np.arange(first_dst, last_dst + 1, dtype=np.int64)
    return np.arange(int(num_nodes), dtype=np.int64)


def make_dir_name(prefix, dataset, seed, **kwargs):
    parts = []
    for key in sorted(kwargs):
        value = kwargs[key]
        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{key}={value}")
    name = "_".join(parts)
    if len(name.encode("utf-8")) > 120:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
        name = f"{name[:96]}_h{digest}"
    return osp.join(prefix, dataset, f"seed{seed}", name)


def save_config(out_dir, config):
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_config(out_dir):
    with open(osp.join(out_dir, "config.json"), "r") as f:
        return json.load(f)


def save_metrics(out_dir, metrics):
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(out_dir):
    with open(osp.join(out_dir, "metrics.json"), "r") as f:
        return json.load(f)


def default_best_params_path(dataset_name, filename="structure_best_params.json"):
    return osp.join(
        osp.dirname(osp.abspath(__file__)),
        "best_hyper_params",
        str(dataset_name),
        filename,
    )


def load_best_hyper_params(dataset_name, path=""):
    path = path or default_best_params_path(dataset_name)
    with open(path, "r") as f:
        payload = json.load(f)
    payload_dataset = payload.get("dataset")
    if payload_dataset is not None and str(payload_dataset) != str(dataset_name):
        raise ValueError(
            f"best params dataset mismatch: file has {payload_dataset}, requested {dataset_name}"
        )
    payload["_path"] = path
    return payload


def best_hyper_param_entries(payload, component, combine_only=False):
    entries = payload.get(component, [])
    if isinstance(entries, dict):
        entries = entries.get("entries", entries.get("params", []))
    if not isinstance(entries, list):
        raise ValueError(f"best params component {component!r} must be a list")
    if combine_only:
        marked = [e for e in entries if isinstance(e, dict) and e.get("combine_use")]
        if marked:
            return marked
    return entries


def best_hyper_param_config(entry):
    if not isinstance(entry, dict):
        raise ValueError("best params entry must be a JSON object")
    params = entry.get("params", entry)
    if not isinstance(params, dict):
        raise ValueError("best params entry params must be a JSON object")
    return params


def is_run_complete(out_dir, modes=("val", "test")):
    if not osp.exists(osp.join(out_dir, "config.json")) or not osp.exists(osp.join(out_dir, "metrics.json")):
        return False
    for mode in modes:
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
            if not osp.exists(osp.join(out_dir, f"{mode}_{suffix}")):
                return False
    return True


def load_scores(out_dir, mode="test"):
    _, _, load_npz, _ = _sparse_tools()
    pos = np.load(osp.join(out_dir, f"{mode}_pos.npy"))
    neg = load_npz(osp.join(out_dir, f"{mode}_neg.npz")).toarray().astype(np.float32)
    lens = np.load(osp.join(out_dir, f"{mode}_valid_lens.npy")).astype(np.int32)
    mask = np.arange(neg.shape[1])[None, :] < lens[:, None]
    return pos.astype(np.float32), np.where(mask, neg, 0.0), mask
