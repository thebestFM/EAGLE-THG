import argparse
import gc
import hashlib
import json
import math
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from single_pipeline.thg_common import (
    SECONDS_PER_DAY,
    add_metric_aliases,
    business_pool,
    format_metrics,
    prefix_metrics,
    split_train_for_prediction,
    stable_hash,
)
from utils import (
    ScoreWriter,
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    describe_loaded_data,
    finalize_metric_sums,
    is_run_complete,
    load_datasets,
    load_metrics,
    save_config,
    save_metrics,
    set_random_seed,
)


def sync_device(device):
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_cuda_peak(device):
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        sync_device(device)
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_mb(device):
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return 0.0
    sync_device(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


class RecentEventStore:
    def __init__(self, num_nodes, topk):
        self.num_nodes = int(num_nodes)
        self.topk = int(topk)
        shape = (self.num_nodes, self.topk)
        self.rel = np.zeros(shape, dtype=np.int32)
        self.neighbor = np.zeros(shape, dtype=np.int32)
        self.event_raw_t = np.zeros(shape, dtype=np.float32)
        self.count = np.zeros(self.num_nodes, dtype=np.int32)
        self.write_pos = np.zeros(self.num_nodes, dtype=np.int32)

    def _write(self, node, rel, neighbor, raw_t):
        if node < 0 or node >= self.num_nodes:
            return
        pos = int(self.write_pos[node])
        self.rel[node, pos] = int(rel)
        self.neighbor[node, pos] = int(neighbor)
        self.event_raw_t[node, pos] = float(raw_t)
        self.write_pos[node] = (pos + 1) % self.topk
        if self.count[node] < self.topk:
            self.count[node] += 1

    def update(self, events, raw_t):
        if self.topk <= 0 or len(events) == 0:
            return
        for user, rel, business in events.astype(np.int64, copy=False):
            self._write(int(user), int(rel), int(business), raw_t)
            self._write(int(business), int(rel) + 6, int(user), raw_t)

    def get_recent(self, nodes, current_raw_t, num_rels, num_nodes):
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        n = len(nodes)
        rels = np.full((n, self.topk), int(num_rels), dtype=np.int64)
        neigh = np.full((n, self.topk), int(num_nodes), dtype=np.int64)
        delta_days = np.zeros((n, self.topk), dtype=np.float32)
        abs_days = np.zeros((n, self.topk), dtype=np.float32)
        mask = np.zeros((n, self.topk), dtype=np.bool_)
        if n == 0:
            return delta_days, abs_days, rels, neigh, mask
        valid_nodes = (nodes >= 0) & (nodes < self.num_nodes)
        if not np.any(valid_nodes):
            return delta_days, abs_days, rels, neigh, mask
        rows = nodes[valid_nodes]
        offsets = np.arange(self.topk, dtype=np.int64)
        idx = (self.write_pos[rows].astype(np.int64)[:, None] - 1 - offsets[None, :]) % self.topk
        row_counts = self.count[rows]
        row_mask = offsets[None, :] < row_counts[:, None]
        out_rows = np.flatnonzero(valid_nodes)
        event_t = self.event_raw_t[rows[:, None], idx]
        rel_values = self.rel[rows[:, None], idx]
        neigh_values = self.neighbor[rows[:, None], idx]
        delta_days[out_rows] = np.maximum(0.0, (float(current_raw_t) - event_t) / SECONDS_PER_DAY) * row_mask
        abs_days[out_rows] = (event_t / SECONDS_PER_DAY) * row_mask
        rel_block = rels[out_rows]
        rel_block[row_mask] = rel_values[row_mask]
        rels[out_rows] = rel_block
        neigh_block = neigh[out_rows]
        valid_neigh = row_mask & (neigh_values >= 0) & (neigh_values < int(num_nodes))
        neigh_block[valid_neigh] = neigh_values[valid_neigh]
        neigh[out_rows] = neigh_block
        mask[out_rows] = row_mask
        return delta_days, abs_days, rels, neigh, mask


class RelationBusinessHistory:
    def __init__(self, num_rels):
        self.pools = [set() for _ in range(int(num_rels))]
        self.arrays = [np.empty(0, dtype=np.int64) for _ in range(int(num_rels))]
        self.dirty = [False for _ in range(int(num_rels))]

    def get(self, rel):
        rel = int(rel)
        if rel < 0 or rel >= len(self.pools):
            return np.empty(0, dtype=np.int64)
        if self.dirty[rel]:
            pool = self.pools[rel]
            self.arrays[rel] = np.asarray(sorted(pool), dtype=np.int64) if pool else np.empty(0, dtype=np.int64)
            self.dirty[rel] = False
        return self.arrays[rel]

    def update(self, events):
        for _, rel, business in events.astype(np.int64, copy=False):
            rel = int(rel)
            before = len(self.pools[rel])
            self.pools[rel].add(int(business))
            if len(self.pools[rel]) != before:
                self.dirty[rel] = True


def timestamp_positive_map(events):
    groups = {}
    for user, rel, business in events.astype(np.int64, copy=False):
        groups.setdefault((int(user), int(rel)), set()).add(int(business))
    return {key: frozenset(values) for key, values in groups.items()}


def sample_without(pool, count, exclude, rng):
    count = int(count)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if len(pool) == 0:
        return np.empty(0, dtype=np.int64)
    exclude = set(int(x) for x in exclude)
    if len(pool) <= count + len(exclude) + 8:
        filtered = np.asarray([int(x) for x in pool if int(x) not in exclude], dtype=np.int64)
        if len(filtered) == 0:
            return np.empty(0, dtype=np.int64)
        if len(filtered) <= count:
            return filtered.copy()
        return rng.choice(filtered, size=count, replace=False)
    selected = []
    seen = set()
    for _ in range(max(64, count * 16)):
        value = int(pool[rng.randint(0, len(pool))])
        if value in exclude or value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=np.int64)


def sample_train_negatives(events, rel_history, dst_pool, num_neg, hard_ratio, rng):
    events = events.astype(np.int64, copy=False)
    num_neg = int(num_neg)
    hard_quota = min(num_neg, max(0, int(math.floor(num_neg * float(hard_ratio)))))
    positives = timestamp_positive_map(events)
    out = np.empty((len(events), num_neg), dtype=np.int64)
    for i, (user, rel, business) in enumerate(events):
        exclude = set(positives.get((int(user), int(rel)), frozenset((int(business),))))
        hard = sample_without(rel_history.get(int(rel)), hard_quota, exclude, rng)
        exclude.update(int(x) for x in hard)
        random = sample_without(dst_pool, num_neg - len(hard), exclude, rng)
        selected = np.concatenate((hard, random)) if len(hard) and len(random) else (hard if len(hard) else random)
        if len(selected) < num_neg:
            fill_pool = dst_pool if len(dst_pool) else np.asarray([int(business)], dtype=np.int64)
            fill = fill_pool[rng.randint(0, len(fill_pool), size=num_neg - len(selected))]
            selected = np.concatenate((selected, fill)) if len(selected) else fill
        out[i] = selected[:num_neg]
    return out


class RecentEncoder(nn.Module):
    def __init__(self, num_nodes, num_rels, node_dim, rel_dim, hidden_dim, dropout):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes + 1, node_dim, padding_idx=num_nodes)
        self.rel_embedding = nn.Embedding(num_rels + 1, rel_dim, padding_idx=num_rels)
        self.time_proj = nn.Sequential(
            nn.Linear(7, rel_dim),
            nn.LayerNorm(rel_dim),
            nn.ReLU(),
        )
        self.event_proj = nn.Sequential(
            nn.Linear(node_dim + rel_dim + rel_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _time_features(self, delta_days, abs_days):
        return torch.stack(
            (
                torch.log1p(delta_days) / 8.0,
                torch.exp(-delta_days / 7.0),
                torch.exp(-delta_days / 30.0),
                torch.exp(-delta_days / 180.0),
                torch.sin(2.0 * math.pi * abs_days / 7.0),
                torch.cos(2.0 * math.pi * abs_days / 7.0),
                torch.sin(2.0 * math.pi * abs_days / 365.25),
            ),
            dim=-1,
        )

    def forward(self, delta_days, abs_days, rel_ids, neighbor_ids, mask):
        rel_feat = self.rel_embedding(rel_ids)
        neigh_feat = self.node_embedding(neighbor_ids)
        time_feat = self.time_proj(self._time_features(delta_days, abs_days))
        x = self.event_proj(torch.cat((rel_feat, neigh_feat, time_feat), dim=-1))
        weights = mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (x * weights).sum(dim=1) / denom
        return self.out_norm(pooled)


class THGTimeModel(nn.Module):
    def __init__(self, num_nodes, num_rels_internal, num_query_rels, node_dim, rel_dim, hidden_dim, dropout):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_rels_internal = int(num_rels_internal)
        self.encoder = RecentEncoder(num_nodes, num_rels_internal, node_dim, rel_dim, hidden_dim, dropout)
        self.query_rel_embedding = nn.Embedding(num_query_rels, rel_dim)
        self.type_embedding = nn.Embedding(2, node_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + rel_dim + node_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def node_type(self, ids, business_first):
        return (ids >= int(business_first)).long()

    def encode_nodes(self, nodes, current_raw_t, store, device):
        delta, abs_days, rels, neigh, mask = store.get_recent(
            nodes,
            current_raw_t,
            self.num_rels_internal,
            self.num_nodes,
        )
        return self.encoder(
            torch.from_numpy(delta).to(device),
            torch.from_numpy(abs_days).to(device),
            torch.from_numpy(rels).to(device),
            torch.from_numpy(neigh).to(device),
            torch.from_numpy(mask).to(device),
        )

    def score_candidates(self, events, candidates, current_raw_t, store, device, business_first):
        events = events.astype(np.int64, copy=False)
        candidates = candidates.astype(np.int64, copy=False)
        batch_size, num_candidates = candidates.shape
        src_nodes = events[:, 0]
        query_rel = torch.from_numpy(events[:, 1].astype(np.int64)).to(device)
        src_repr = self.encode_nodes(src_nodes, current_raw_t, store, device)
        dst_repr = self.encode_nodes(candidates.reshape(-1), current_raw_t, store, device)
        src_repr = src_repr.unsqueeze(1).expand(batch_size, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        query_rel_feat = self.query_rel_embedding(query_rel).unsqueeze(1).expand(batch_size, num_candidates, -1)
        query_rel_feat = query_rel_feat.reshape(batch_size * num_candidates, -1)
        src_ids = torch.from_numpy(np.repeat(src_nodes.reshape(-1, 1), num_candidates, axis=1).reshape(-1)).to(device)
        dst_ids = torch.from_numpy(candidates.reshape(-1)).to(device)
        src_type = self.type_embedding(self.node_type(src_ids, business_first))
        dst_type = self.type_embedding(self.node_type(dst_ids, business_first))
        scores = self.scorer(torch.cat((src_repr, dst_repr, query_rel_feat, src_type, dst_type), dim=-1)).squeeze(-1)
        return scores.view(batch_size, num_candidates)


def make_store(args, data):
    return RecentEventStore(data["num_nodes"], int(args.topk))


def update_store(store, snapshot_list):
    for events, _, raw_t in snapshot_list:
        store.update(events, raw_t)
    return store


def build_model(args, data, device):
    return THGTimeModel(
        num_nodes=data["num_nodes"],
        num_rels_internal=data["num_rels"] * 2,
        num_query_rels=data["num_rels"],
        node_dim=args.node_dim,
        rel_dim=args.rel_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)


def train_one_epoch(model, train_list, data, args, device, rng):
    model.train()
    store = make_store(args, data)
    rel_history = RelationBusinessHistory(data["num_rels"])
    dst_pool = business_pool(data)
    total_loss = 0.0
    total_rows = 0
    start = time.time()
    for events, _, raw_t in tqdm(train_list, desc="thg_time_train", leave=False):
        rel_history.update(events)
        for batch_start in range(0, len(events), int(args.batch_size)):
            batch = events[batch_start : batch_start + int(args.batch_size)]
            neg = sample_train_negatives(batch, rel_history, dst_pool, args.train_num_neg, args.hard_neg_ratio, rng)
            candidates = np.concatenate((batch[:, 2:3], neg), axis=1)
            scores = model.score_candidates(batch, candidates, raw_t, store, device, data["business_first_id"])
            if args.train_loss == "ce":
                labels = torch.zeros(len(batch), dtype=torch.long, device=device)
                loss = F.cross_entropy(scores / float(args.temperature), labels)
            else:
                pos = scores[:, 0]
                neg_max = scores[:, 1:].max(dim=1).values
                target = torch.ones_like(pos)
                loss = F.margin_ranking_loss(pos, neg_max, target, margin=float(args.rank_margin))
            args._optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            args._optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_rows += len(batch)
        store.update(events, raw_t)
    return total_loss / max(total_rows, 1), float(time.time() - start)


@torch.no_grad()
def evaluate_split(model, snapshot_list, store, data, args, device, mode, out_dir=None, write_scores=False):
    model.eval()
    writer = ScoreWriter(out_dir, mode) if write_scores else None
    sums = {}
    start = time.time()
    score_time = 0.0
    for events, _, raw_t in tqdm(snapshot_list, desc=f"thg_time_{mode}", leave=False):
        for batch, neg_arr, neg_mask in collect_eval_batch(events, raw_t, data["negative_sampler"], mode, args.eval_batch_size):
            pos_scores = []
            neg_scores = np.zeros(neg_arr.shape, dtype=np.float32)
            chunk = max(1, int(args.eval_neg_chunk))
            t_score = time.time()
            pos = model.score_candidates(batch, batch[:, 2:3], raw_t, store, device, data["business_first_id"])
            pos_scores = pos.detach().cpu().numpy().astype(np.float32)
            for start_col in range(0, neg_arr.shape[1], chunk):
                end_col = min(start_col + chunk, neg_arr.shape[1])
                neg_chunk = neg_arr[:, start_col:end_col].copy()
                neg_chunk[neg_chunk < 0] = data["business_first_id"]
                scores = model.score_candidates(batch, neg_chunk, raw_t, store, device, data["business_first_id"])
                neg_scores[:, start_col:end_col] = scores.detach().cpu().numpy().astype(np.float32)
            sync_device(device)
            score_time += time.time() - t_score
            batch_sums = compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask)
            add_metric_sums(sums, batch_sums)
            if writer is not None:
                writer.write_batch(pos_scores, neg_scores, neg_mask)
        store.update(events, raw_t)
    if writer is not None:
        writer.close()
    metrics = add_metric_aliases(finalize_metric_sums(sums))
    metrics["profile"] = {"eval_time_sec": float(time.time() - start), "score_time_sec": float(score_time)}
    print(f"[THG-Time] {mode} {format_metrics(metrics)} time={metrics['profile']['eval_time_sec']:.1f}s", flush=True)
    return metrics


def train_phase(model, train_list, data, args, device, num_epochs, select_with_val, best_path=None):
    best_score = -float("inf")
    best_epoch = 0
    best_metrics = {}
    bad_rounds = 0
    train_time = 0.0
    peak = 0.0
    rng = np.random.RandomState(int(args.seed))
    args._optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    for epoch in range(1, int(num_epochs) + 1):
        reset_cuda_peak(device)
        loss, elapsed = train_one_epoch(model, train_list, data, args, device, rng)
        train_time += elapsed
        peak = max(peak, cuda_peak_mb(device))
        if not select_with_val:
            best_epoch = epoch
            print(f"[THG-Time] epoch={epoch} loss={loss:.5f} train_time={elapsed:.1f}s", flush=True)
            continue
        val_store = make_store(args, data)
        update_store(val_store, data["train_list"])
        val_metrics = evaluate_split(model, data["val_list"], val_store, data, args, device, "val", write_scores=False)
        score = float(val_metrics.get(args.selection_metric, val_metrics["mrr"]))
        print(
            f"[THG-Time] epoch={epoch} loss={loss:.5f} "
            f"select_{args.selection_metric}={score:.5f} train_time={elapsed:.1f}s",
            flush=True,
        )
        if score > best_score + float(args.tolerance):
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            bad_rounds = 0
            if best_path is not None:
                torch.save(model.state_dict(), best_path)
        else:
            bad_rounds += 1
            if bad_rounds >= int(args.patience):
                break
    if select_with_val and best_path is not None and osp.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    delattr(args, "_optimizer")
    return {
        "model": model,
        "best_score": float(best_score if select_with_val else 0.0),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "train_time_sec": float(train_time),
        "train_peak_alloc_mb": float(peak),
    }


def run_params(args):
    return {
        "topk": args.topk,
        "node_dim": args.node_dim,
        "rel_dim": args.rel_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "train_num_neg": args.train_num_neg,
        "hard_neg_ratio": args.hard_neg_ratio,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "selection_metric": args.selection_metric,
        "ns_q": args.ns_q,
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
    }


def get_out_dir(args):
    payload = {"dataset": args.dataset, "seed": args.seed, "params": run_params(args)}
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    name = (
        f"r{h}_topk{args.topk}_hd{args.hidden_dim}_nd{args.node_dim}"
        f"_bs{args.batch_size}_ebs{args.eval_batch_size}_nsq{args.ns_q}"
        f"_tpr{args.train_predict_ratio:g}"
    )
    return osp.join("results_thg_time", args.dataset, f"seed{args.seed}", name)


def score_modes(args):
    modes = ("train", "val") if float(args.train_predict_ratio) > 0.0 else ("val",)
    if getattr(args, "eval_test", True):
        modes = modes + ("test",)
    return modes


def validate_args(args):
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or positive")
    if not 0.0 <= float(args.train_predict_ratio) < 1.0:
        raise ValueError("--train_predict_ratio must be in [0,1)")
    if int(args.topk) <= 0 or int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0:
        raise ValueError("topk/batch sizes must be positive")
    if int(args.train_num_neg) <= 0 or int(args.eval_neg_chunk) <= 0:
        raise ValueError("negative counts/chunks must be positive")
    args.hard_neg_ratio = min(1.0, max(0.0, float(args.hard_neg_ratio)))


def main(args):
    validate_args(args)
    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    out_dir = get_out_dir(args)
    if is_run_complete(out_dir, score_modes(args)) and not getattr(args, "force", False):
        print(f"[THG-Time] already complete: {out_dir}", flush=True)
        return load_metrics(out_dir)

    data = load_datasets(args.dataset, q=args.ns_q, load_train_ratio=args.train_predict_ratio, ns_seed=args.ns_seed)
    if not data.get("is_thg"):
        raise ValueError("THG time component requires a Yelp THG dataset")
    describe_loaded_data(data, prefix="[THG-Time]")
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args, data, device)
    best_path = osp.join(out_dir, "best_model.pt")
    full_result = train_phase(
        model,
        data["train_list"],
        data,
        args,
        device,
        num_epochs=args.num_epochs,
        select_with_val=True,
        best_path=best_path,
    )
    model = full_result["model"]
    best_epoch = max(1, int(full_result["best_epoch"]))
    train_prefix, train_predict = split_train_for_prediction(data)

    train_metrics = None
    oof_result = None
    if train_predict:
        print(f"[THG-Time] OOF prefix model epochs={best_epoch} prefix_ts={len(train_prefix)}", flush=True)
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_random_seed(args.seed)
        oof_model = build_model(args, data, device)
        oof_result = train_phase(oof_model, train_prefix, data, args, device, best_epoch, select_with_val=False)
        oof_store = make_store(args, data)
        update_store(oof_store, train_prefix)
        train_metrics = evaluate_split(
            oof_result["model"],
            train_predict,
            oof_store,
            data,
            args,
            device,
            "train",
            out_dir=out_dir,
            write_scores=True,
        )
        del oof_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(device)

    val_store = make_store(args, data)
    update_store(val_store, data["train_list"])
    val_metrics = evaluate_split(model, data["val_list"], val_store, data, args, device, "val", out_dir, True)
    test_metrics = None
    test_peak = 0.0
    if getattr(args, "eval_test", True):
        reset_cuda_peak(device)
        test_store = make_store(args, data)
        update_store(test_store, data["train_list"])
        update_store(test_store, data["val_list"])
        test_metrics = evaluate_split(model, data["test_list"], test_store, data, args, device, "test", out_dir, True)
        test_peak = cuda_peak_mb(device)

    metrics = {
        "format": "thg_time_scores_v1",
        "score_protocol": "train_oof_prefix_valtest_full_train",
        "best_model_path": best_path,
        "best_epoch": int(best_epoch),
        "best_val_selection": float(full_result["best_score"]),
        "train_time_sec": float(full_result["train_time_sec"] + (oof_result or {}).get("train_time_sec", 0.0)),
        "train_peak_alloc_mb": float(max(full_result["train_peak_alloc_mb"], (oof_result or {}).get("train_peak_alloc_mb", 0.0))),
        "infer_peak_alloc_mb": float(test_peak),
    }
    if train_metrics is not None:
        metrics.update(prefix_metrics("train", train_metrics))
    metrics.update(prefix_metrics("val", val_metrics))
    if test_metrics is not None:
        metrics.update(prefix_metrics("test", test_metrics))
    save_config(out_dir, {**vars(args), "out_dir": out_dir, "run_params": run_params(args)})
    save_metrics(out_dir, metrics)
    print(f"[THG-Time] saved -> {out_dir}", flush=True)
    return metrics


def load_args():
    parser = argparse.ArgumentParser("Run the THG time component.")
    parser.add_argument("--dataset", type=str, default="Yelp-NOLA", choices=("Yelp-NOLA", "Yelp-PHL", "Yelp-TPA"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--node_dim", type=int, default=64)
    parser.add_argument("--rel_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_neg_chunk", type=int, default=512)
    parser.add_argument("--train_num_neg", type=int, default=8)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--train_loss", type=str, default="margin", choices=("margin", "ce"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--selection_metric", type=str, default="hit10")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no_eval_test", dest="eval_test", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(load_args())
