import gc
import heapq
import json
import math
import os
import os.path as osp
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

import train_new_structure as tns
from utils import (
    HIT_KS,
    ScoreStore,
    ScoreWriter,
    add_metric_sums,
    add_rank_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    dense_rank,
    finalize_metric_sums,
    inverse_aug,
    ranking_metric_key,
)


EPS = 1e-12
COMPONENT_SCORE_NAMES = ("dsh", "dmh", "shared", "direct", "structure_raw", "b", "base")


@dataclass(frozen=True)
class BConfig:
    mode: str = "continuous"
    binary_unseen: float = 0.0
    continuous_alpha: float = 0.0001


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def split_snapshots(data, split):
    if split == "train":
        return data["train_list"][data["train_predict_start_idx"] :]
    if split == "val":
        return data["val_list"]
    if split == "test":
        return data["test_list"]
    raise ValueError(split)


def events_for_update(events, data, args, is_train=False):
    events_f64 = np.ascontiguousarray(events, dtype=np.float64)
    if not is_train and data["is_tgb"] and not bool(getattr(args, "close_update_backward", False)):
        return inverse_aug(events_f64, data["num_rels_raw"], data["num_rels"])
    return events_f64


def metric_value(metrics, metric):
    return float(metrics[ranking_metric_key(metric, strict=True)])


def format_metrics(metrics):
    parts = [
        f"mrr={metrics['mrr_strict']:.5f}",
        f"hr1={metrics['hit@1_strict']:.5f}",
        f"hr10={metrics['hit@10_strict']:.5f}",
    ]
    return " ".join(parts)


def max_norm(scores, valid):
    max_score = np.max(np.where(valid, scores, 0.0), axis=1, keepdims=True)
    denom = np.where(max_score > 0.0, max_score, 1.0)
    return np.where(valid, scores / denom, 0.0).astype(np.float32, copy=False)


def zscore_by_query(scores, valid):
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(valid, scores, 0.0), axis=1, keepdims=True) / count
    var = np.sum(np.where(valid, (scores - mean) ** 2, 0.0), axis=1, keepdims=True) / count
    std = np.where(var <= EPS, 1.0, np.sqrt(var))
    return np.where(valid, (scores - mean) / std, 0.0).astype(np.float32, copy=False)


def minmax_by_query(scores, valid):
    low = np.min(np.where(valid, scores, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(valid, scores, -np.inf), axis=1, keepdims=True)
    denom = np.maximum(high - low, EPS)
    out = (scores - low) / denom
    return np.where(valid, out, 0.0).astype(np.float32, copy=False)


def candidate_rank_pair(scores, valid):
    pos = scores[:, :1]
    neg = scores[:, 1:]
    mask = valid[:, 1:]
    loose = 1 + np.sum((neg > pos) & mask, axis=1)
    strict = 1 + np.sum((neg >= pos) & mask, axis=1)
    avg = (loose.astype(np.float64) + strict.astype(np.float64)) * 0.5
    return loose.astype(np.int64), strict.astype(np.int64), avg


def check_finite_scores(label, scores, valid):
    bad = int(np.size(scores[valid]) - np.sum(np.isfinite(scores[valid])))
    if bad:
        raise RuntimeError(f"{label} has {bad} non-finite valid scores")


def event_delta(events, num_nodes):
    if len(events) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    rels = events[:, 1].astype(np.int64, copy=False)
    objs = events[:, 2].astype(np.int64, copy=False)
    flat = rels * int(num_nodes) + objs
    unique, counts = np.unique(flat, return_counts=True)
    return (
        (unique // int(num_nodes)).astype(np.int64),
        (unique % int(num_nodes)).astype(np.int64),
        counts.astype(np.float32),
    )


class BTimeline:
    def __init__(self, num_rels, num_nodes):
        self.num_rels = int(num_rels)
        self.num_nodes = int(num_nodes)
        self.counts = {}
        self.counts_r = np.zeros(self.num_rels, dtype=np.float32)

    def update(self, events):
        rels, objs, vals = event_delta(events, self.num_nodes)
        for r, o, v in zip(rels, objs, vals):
            key = int(r) * self.num_nodes + int(o)
            self.counts[key] = self.counts.get(key, 0.0) + float(v)
        if len(rels):
            self.counts_r += np.bincount(rels, weights=vals, minlength=self.num_rels).astype(np.float32)

    def count_batch(self, batch_data, neg_arr, width):
        rels = batch_data[:, 1].astype(np.int64)
        pos_obj = batch_data[:, 2].astype(np.int64)
        neg_obj = np.where(neg_arr[:, :width] < 0, 0, neg_arr[:, :width]).astype(np.int64)
        pos_counts = np.zeros(len(batch_data), dtype=np.float32)
        neg_counts = np.zeros((len(batch_data), width), dtype=np.float32)
        for i, r in enumerate(rels):
            pos_counts[i] = self.counts.get(int(r) * self.num_nodes + int(pos_obj[i]), 0.0)
            for j in range(width):
                if neg_arr[i, j] >= 0:
                    neg_counts[i, j] = self.counts.get(int(r) * self.num_nodes + int(neg_obj[i, j]), 0.0)
        return rels, pos_counts, neg_counts

    def score_batch(self, rels, pos_counts, neg_counts, b_cfg):
        if b_cfg.mode == "binary":
            pos = np.where(pos_counts > 0, 1.0, b_cfg.binary_unseen).astype(np.float32).reshape(-1, 1)
            neg = np.where(neg_counts > 0, 1.0, b_cfg.binary_unseen).astype(np.float32)
            return pos, neg
        alpha = float(b_cfg.continuous_alpha)
        denom = np.maximum(self.counts_r[rels] + alpha * self.num_nodes, EPS)
        pos = ((pos_counts + alpha) / denom).astype(np.float32).reshape(-1, 1)
        neg = ((neg_counts + alpha) / denom.reshape(-1, 1)).astype(np.float32)
        return pos, neg


class CausalHistory:
    def __init__(self, num_nodes, num_rels):
        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.source_count = np.zeros(self.num_nodes, dtype=np.float32)
        self.tail_count = np.zeros(self.num_nodes, dtype=np.float32)
        self.incident_count = np.zeros(self.num_nodes, dtype=np.float32)
        self.rel_count = np.zeros(self.num_rels, dtype=np.float32)
        self.source_rel = {}
        self.sr_tail = {}
        self.source_tail = {}

    def update(self, events):
        for s, r, o in events[:, :3].astype(np.int64, copy=False):
            s = int(s)
            r = int(r)
            o = int(o)
            if 0 <= s < self.num_nodes:
                self.source_count[s] += 1.0
                self.incident_count[s] += 1.0
            if 0 <= o < self.num_nodes:
                self.tail_count[o] += 1.0
                self.incident_count[o] += 1.0
            if 0 <= r < self.num_rels:
                self.rel_count[r] += 1.0
            sr = (s, r)
            self.source_rel[sr] = self.source_rel.get(sr, 0.0) + 1.0
            self.sr_tail.setdefault(sr, {})[o] = self.sr_tail.setdefault(sr, {}).get(o, 0) + 1
            self.source_tail.setdefault(s, {})[o] = self.source_tail.setdefault(s, {}).get(o, 0) + 1

    def features(self, sources, rels, cand_ids, rel_tail_counts):
        bsz, width = cand_ids.shape
        safe_o = np.where((cand_ids >= 0) & (cand_ids < self.num_nodes), cand_ids, 0)
        safe_s = np.where((sources >= 0) & (sources < self.num_nodes), sources, 0)
        safe_r = np.where((rels >= 0) & (rels < self.num_rels), rels, 0)
        source_count = np.repeat(self.source_count[safe_s].reshape(-1, 1), width, axis=1)
        rel_count = np.repeat(self.rel_count[safe_r].reshape(-1, 1), width, axis=1)
        source_rel = np.zeros((bsz, width), dtype=np.float32)
        exact_sro = np.zeros((bsz, width), dtype=np.float32)
        source_tail = np.zeros((bsz, width), dtype=np.float32)
        for i in range(bsz):
            source_rel[i, :] = self.source_rel.get((int(sources[i]), int(rels[i])), 0.0)
            sr_counter = self.sr_tail.get((int(sources[i]), int(rels[i])), {})
            st_counter = self.source_tail.get(int(sources[i]), {})
            exact_sro[i, :] = [sr_counter.get(int(o), 0) for o in cand_ids[i]]
            source_tail[i, :] = [st_counter.get(int(o), 0) for o in cand_ids[i]]
        invalid = (cand_ids < 0) | (cand_ids >= self.num_nodes)
        arrays = [
            self.tail_count[safe_o],
            self.incident_count[safe_o],
            rel_tail_counts,
            source_count,
            rel_count,
            source_rel,
            exact_sro,
            source_tail,
        ]
        return [np.where(invalid, 0.0, x).astype(np.float32, copy=False) for x in arrays]


class StructureComponentRuntime:
    def __init__(self, data, args, device):
        max_t_norm = tns.estimate_max_t_norm(data)
        self.direct = tns.DirectSingleHopScorer(
            data["num_rels"],
            decay_direct=float(args.decay_direct),
            max_time_span=max_t_norm,
            log_bucket_stats=bool(getattr(args, "dsh_log_bucket_stats", False)),
        )
        self.predictor, self.semantic_updater, self.logic_updater = tns.build_runtime(
            args,
            data["num_nodes"],
            data["num_rels"],
            device,
        )
        self.args = args

    def ensure_ready_for_prediction(self):
        self.predictor.ensure_M_sim(self.semantic_updater)

    def predict_parts(self, batch_data, neg_arr):
        self.ensure_ready_for_prediction()
        batch_i64 = np.ascontiguousarray(batch_data[:, :3].astype(np.int64, copy=False))
        neg_i64 = np.ascontiguousarray(neg_arr.astype(np.int64, copy=False))
        dsh_pos, dsh_neg = self.direct.predict_batch(batch_i64, neg_i64)
        dmh_pos, dmh_neg, shared_pos, shared_neg = predict_dmh_shared_parts(self.predictor, batch_i64, neg_i64)
        final_direct_pos, final_direct_neg = tns._combine_direct_scores(
            dsh_pos, dsh_neg, dmh_pos, dmh_neg, float(self.args.direct_single_hop)
        )
        if float(self.args.gamma) > 0.0 and int(getattr(self.args, "top_direct", -1)) >= 0:
            pos_mask, neg_mask = tns._top_direct_masks(
                final_direct_pos,
                final_direct_neg,
                neg_i64,
                int(self.args.top_direct),
            )
            shared_pos = np.where(pos_mask, shared_pos, 0.0).astype(np.float32, copy=False)
            shared_neg = np.where(neg_mask, shared_neg, 0.0).astype(np.float32, copy=False)
        final_pos = final_direct_pos + float(self.args.gamma) * shared_pos
        final_neg = final_direct_neg + float(self.args.gamma) * shared_neg
        return {
            "dsh": (dsh_pos, dsh_neg),
            "dmh": (dmh_pos, dmh_neg),
            "shared": (shared_pos, shared_neg),
            "direct": (final_direct_pos, final_direct_neg),
            "structure_raw": (final_pos, final_neg),
        }

    def update(self, events, t_norm):
        events_f64 = np.asarray(events, dtype=np.float64)
        tns.update_runtime(
            self.predictor,
            self.direct,
            events_f64,
            t_norm,
            self.semantic_updater,
            self.logic_updater,
        )


def predict_dmh_shared_parts(predictor, batch_i64, neg_i64):
    if hasattr(predictor, "_predict_parts"):
        return predictor._predict_parts(batch_i64, neg_i64, True)
    if getattr(predictor, "M_sim_np", None) is None:
        raise RuntimeError("M_sim must be synced before prediction")
    pos_direct = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
    neg_direct = np.zeros(neg_i64.shape, dtype=np.float32)
    pos_shared = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
    neg_shared = np.zeros(neg_i64.shape, dtype=np.float32)
    use_top_direct = False
    pos_mask = np.empty((0, 0), dtype=np.bool_)
    neg_mask = np.empty((0, 0), dtype=np.bool_)
    if isinstance(predictor, tns.FastPerRelSourceJoinPredictor):
        tns._fast_perrel_predict_parts_batch(
            predictor.entry_keys,
            predictor.rel_keys,
            predictor.rel_scores,
            predictor.entry_lens,
            predictor.rel_lens,
            predictor.M_sim_np,
            batch_i64,
            neg_i64,
            predictor.shared_w,
            predictor.top_share,
            pos_direct,
            neg_direct,
            pos_shared,
            neg_shared,
            use_top_direct,
            pos_mask,
            neg_mask,
        )
    else:
        tns._fast_tag_predict_parts_batch(
            predictor.keys,
            predictor.scores,
            predictor.tags,
            predictor.lens,
            predictor.M_sim_np,
            batch_i64,
            neg_i64,
            predictor.shared_w,
            predictor.top_share,
            pos_direct,
            neg_direct,
            pos_shared,
            neg_shared,
            use_top_direct,
            pos_mask,
            neg_mask,
        )
    return pos_direct, neg_direct, pos_shared, neg_shared


def init_stream_state(data, split, args, device):
    runtime = StructureComponentRuntime(data, args, device)
    timeline = BTimeline(data["num_rels"], data["num_nodes"])
    history = CausalHistory(data["num_nodes"], data["num_rels"])
    warmup = []
    if split == "train":
        warmup = [(snap, True) for snap in data["train_list"][: data["train_predict_start_idx"]]]
    elif split == "val":
        warmup = [(snap, True) for snap in data["train_list"]]
    elif split == "test":
        warmup = [(snap, True) for snap in data["train_list"]]
        warmup.extend((snap, False) for snap in data["val_list"])
    for (events, t_norm, _), is_train_update in warmup:
        update_events = events_for_update(events, data, args, is_train=is_train_update)
        runtime.update(update_events, t_norm)
        timeline.update(update_events)
        history.update(update_events)
    return runtime, timeline, history


def init_timeline_history_state(data, split, args):
    timeline = BTimeline(data["num_rels"], data["num_nodes"])
    history = CausalHistory(data["num_nodes"], data["num_rels"])
    warmup = []
    if split == "train":
        warmup = [(snap, True) for snap in data["train_list"][: data["train_predict_start_idx"]]]
    elif split == "val":
        warmup = [(snap, True) for snap in data["train_list"]]
    elif split == "test":
        warmup = [(snap, True) for snap in data["train_list"]]
        warmup.extend((snap, False) for snap in data["val_list"])
    for (events, _, _), is_train_update in warmup:
        update_events = events_for_update(events, data, args, is_train=is_train_update)
        timeline.update(update_events)
        history.update(update_events)
    return timeline, history


def component_score_dict(parts, b_pos, b_neg, valid):
    scores = {}
    for name, (pos, neg) in parts.items():
        scores[name] = np.concatenate((pos, neg), axis=1).astype(np.float32, copy=False)
    scores["b"] = np.concatenate((b_pos, b_neg), axis=1).astype(np.float32, copy=False)
    scores["base"] = structure_base_scores(scores, valid)
    for key, val in scores.items():
        check_finite_scores(f"component {key}", val, valid)
    return scores


def structure_base_scores(scores, valid):
    keys = ("dsh", "dmh", "shared", "b")
    total = np.zeros_like(scores["dsh"], dtype=np.float32)
    for key in keys:
        total += max_norm(scores[key], valid)
    return (total / float(len(keys))).astype(np.float32, copy=False)


class StructureFeatureBuilder:
    def __init__(self, num_rels):
        self.num_rels = int(num_rels)
        self.feature_names = []
        self.categorical_names = []
        self._init_names()

    def _add(self, name, categorical=False):
        self.feature_names.append(name)
        if categorical:
            self.categorical_names.append(name)

    @property
    def categorical_indices(self):
        positions = {name: i for i, name in enumerate(self.feature_names)}
        return [positions[name] for name in self.categorical_names]

    def _init_names(self):
        for prefix in ("base", "dsh", "dmh", "shared", "direct", "structure_raw", "b"):
            self._add(f"{prefix}_score")
            self._add(f"{prefix}_log1p")
            self._add(f"{prefix}_max_norm")
            self._add(f"{prefix}_z")
            self._add(f"{prefix}_rank_log")
            self._add(f"{prefix}_rank_recip")
        for name in (
            "dsh_minus_dmh",
            "direct_minus_shared",
            "structure_minus_b",
            "dsh_times_dmh",
            "direct_times_shared",
            "structure_times_b",
            "component_mean",
            "component_max",
            "component_std",
            "b_count_log1p",
            "b_seen",
            "tail_count_log1p",
            "incident_count_log1p",
            "rel_tail_count_log1p",
            "source_count_log1p",
            "rel_count_log1p",
            "source_rel_log1p",
            "exact_sro_log1p",
            "source_tail_log1p",
            "relation_is_inverse",
            "candidate_is_source",
        ):
            self._add(name)
        self._add("relation_id", categorical=True)
        self._add("source_id", categorical=True)
        self._add("candidate_id", categorical=True)

    def make(self, batch_data, cand_ids, valid, scores, b_counts, history):
        sources = batch_data[:, 0].astype(np.int64)
        rels = batch_data[:, 1].astype(np.int64)
        width = cand_ids.shape[1]
        features = []
        for prefix in ("base", "dsh", "dmh", "shared", "direct", "structure_raw", "b"):
            sc = np.where(valid, scores[prefix], 0.0).astype(np.float32, copy=False)
            ranks = dense_rank(sc, valid).astype(np.float32)
            features.extend(
                [
                    sc,
                    np.log1p(np.maximum(sc, 0.0)).astype(np.float32, copy=False),
                    max_norm(sc, valid),
                    zscore_by_query(sc, valid),
                    np.log1p(ranks).astype(np.float32, copy=False),
                    (1.0 / np.maximum(ranks, 1.0)).astype(np.float32, copy=False),
                ]
            )
        dsh = scores["dsh"]
        dmh = scores["dmh"]
        shared = scores["shared"]
        direct = scores["direct"]
        structure = scores["structure_raw"]
        b = scores["b"]
        stack = np.stack([dsh, dmh, shared, direct, structure, b], axis=2)
        features.extend(
            [
                (dsh - dmh).astype(np.float32, copy=False),
                (direct - shared).astype(np.float32, copy=False),
                (structure - b).astype(np.float32, copy=False),
                (dsh * dmh).astype(np.float32, copy=False),
                (direct * shared).astype(np.float32, copy=False),
                (structure * b).astype(np.float32, copy=False),
                np.mean(stack, axis=2).astype(np.float32, copy=False),
                np.max(stack, axis=2).astype(np.float32, copy=False),
                np.std(stack, axis=2).astype(np.float32, copy=False),
                np.log1p(b_counts).astype(np.float32, copy=False),
                (b_counts > 0).astype(np.float32, copy=False),
            ]
        )
        for arr in history.features(sources, rels, cand_ids, b_counts):
            features.append(np.log1p(arr).astype(np.float32, copy=False))
        rel_matrix = np.repeat(rels.reshape(-1, 1), width, axis=1)
        src_matrix = np.repeat(sources.reshape(-1, 1), width, axis=1)
        features.append((rels.reshape(-1, 1) >= self.num_rels // 2).repeat(width, axis=1).astype(np.float32))
        features.append((cand_ids == src_matrix).astype(np.float32, copy=False))
        features.append(rel_matrix.astype(np.float32, copy=False))
        features.append(src_matrix.astype(np.float32, copy=False))
        features.append(np.where(valid, cand_ids, 0).astype(np.float32, copy=False))
        cube = np.stack(features, axis=2).astype(np.float32, copy=False)
        bad = int(np.size(cube) - np.sum(np.isfinite(cube)))
        if bad:
            raise RuntimeError(f"structure feature cube has {bad} non-finite values")
        return cube


def selected_from_base(base_scores, valid, topk):
    selected = np.zeros_like(valid, dtype=bool)
    selected[:, 0] = True
    if int(topk) < 0:
        selected |= valid
    else:
        selected |= (dense_rank(base_scores, valid) <= int(topk) + 1) & valid
    return selected & valid


def iter_structure_blocks(data, split, args, feature_builder, device):
    runtime, timeline, history = init_stream_state(data, split, args, device)
    neg_sampler = data["negative_sampler"]
    snapshots = split_snapshots(data, split)
    for events, t_norm, t_orig in snapshots:
        for batch_data, neg_arr, neg_mask in collect_eval_batch(events, t_orig, neg_sampler, split, args.query_batch_size):
            width = int(neg_arr.shape[1])
            parts = runtime.predict_parts(batch_data, neg_arr)
            rels, pos_counts, neg_counts = timeline.count_batch(batch_data, neg_arr, width)
            b_pos, b_neg = timeline.score_batch(rels, pos_counts, neg_counts, args.b_cfg)
            valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), neg_mask[:, :width]), axis=1)
            cand_ids = np.concatenate((batch_data[:, 2:3], neg_arr[:, :width]), axis=1)
            cand_ids = np.where(valid, cand_ids, -1).astype(np.int64, copy=False)
            b_counts = np.concatenate((pos_counts.reshape(-1, 1), neg_counts), axis=1).astype(np.float32, copy=False)
            scores = component_score_dict(parts, b_pos, b_neg, valid)
            cube = feature_builder.make(batch_data, cand_ids, valid, scores, b_counts, history)
            yield SimpleNamespace(
                batch_data=batch_data,
                valid=valid,
                scores=scores,
                cand_ids=cand_ids,
                cube=cube,
                t_norm=t_norm,
                t_orig=t_orig,
            )
        update_events = events_for_update(events, data, args, is_train=(split == "train"))
        runtime.update(update_events, t_norm)
        timeline.update(update_events)
        history.update(update_events)


def iter_component_blocks(data, split, args, device):
    runtime, timeline, history = init_stream_state(data, split, args, device)
    neg_sampler = data["negative_sampler"]
    snapshots = split_snapshots(data, split)
    for events, t_norm, t_orig in snapshots:
        for batch_data, neg_arr, neg_mask in collect_eval_batch(events, t_orig, neg_sampler, split, args.query_batch_size):
            width = int(neg_arr.shape[1])
            parts = runtime.predict_parts(batch_data, neg_arr)
            rels, pos_counts, neg_counts = timeline.count_batch(batch_data, neg_arr, width)
            b_pos, b_neg = timeline.score_batch(rels, pos_counts, neg_counts, args.b_cfg)
            valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), neg_mask[:, :width]), axis=1)
            cand_ids = np.concatenate((batch_data[:, 2:3], neg_arr[:, :width]), axis=1)
            cand_ids = np.where(valid, cand_ids, -1).astype(np.int64, copy=False)
            b_counts = np.concatenate((pos_counts.reshape(-1, 1), neg_counts), axis=1).astype(np.float32, copy=False)
            scores = component_score_dict(parts, b_pos, b_neg, valid)
            history_values = history.features(
                batch_data[:, 0].astype(np.int64, copy=False),
                batch_data[:, 1].astype(np.int64, copy=False),
                cand_ids,
                b_counts,
            )
            yield SimpleNamespace(
                batch_data=batch_data,
                valid=valid,
                scores=scores,
                cand_ids=cand_ids,
                b_counts=b_counts,
                history_values=history_values,
                t_norm=t_norm,
                t_orig=t_orig,
            )
        update_events = events_for_update(events, data, args, is_train=(split == "train"))
        runtime.update(update_events, t_norm)
        timeline.update(update_events)
        history.update(update_events)


def build_structure_matrix(data, split, args, feature_builder, device, topk):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0
    for block in iter_structure_blocks(data, split, args, feature_builder, device):
        selected = selected_from_base(block.scores["base"], block.valid, topk)
        for row in range(selected.shape[0]):
            cols = np.flatnonzero(selected[row])
            if len(cols) <= 1:
                continue
            pos = np.flatnonzero(cols == 0)
            if len(pos) != 1:
                raise RuntimeError(f"{split} structure matrix row missing positive")
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[int(pos[0])] = 1.0
            X_parts.append(block.cube[row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)
    if not X_parts:
        raise RuntimeError(f"no structure rows built for split={split}")
    positives = int(np.sum(np.concatenate(y_parts) > 0.0))
    if positives != queries:
        raise RuntimeError(f"structure matrix expected one positive per query, got positives={positives}, queries={queries}")
    nonfinite = int(sum(np.size(x) - np.sum(np.isfinite(x)) for x in X_parts))
    if nonfinite:
        raise RuntimeError(f"structure matrix has {nonfinite} non-finite feature values")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {"split": split, "queries": int(queries), "rows": int(rows), "topk": int(topk)},
    )


def component_score_root(out_dir, struct_id):
    return osp.join(out_dir, "component_scores", str(struct_id))


def save_component_score_stores(data, args, device, out_dir, struct_id, splits=None, compute_metrics=True):
    root = ensure_dir(component_score_root(out_dir, struct_id))
    if splits is None:
        splits = ("train", "val", "test")
    splits = tuple(dict.fromkeys(str(s) for s in splits))
    compute_metrics = bool(compute_metrics)
    metrics_path = osp.join(root, "metrics.json")
    expected = [
        osp.join(root, split, name, f"{split}_{suffix}")
        for split in splits
        for name in COMPONENT_SCORE_NAMES
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json")
    ]
    if osp.isfile(metrics_path) and all(osp.isfile(path) for path in expected):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        cached_splits = set(metrics.get("splits", {}).keys())
        if set(splits).issubset(cached_splits):
            print(f"[ComponentScores] cache hit struct={struct_id} splits={','.join(splits)} -> {root}", flush=True)
            return root, metrics
    metrics = {
        "format": "new_structure_component_scores_v1",
        "struct_id": str(struct_id),
        "compute_metrics": compute_metrics,
        "splits": {},
    }
    for split in splits:
        split_dir = ensure_dir(osp.join(root, split))
        writers = {
            name: ScoreWriter(ensure_dir(osp.join(split_dir, name)), split)
            for name in COMPONENT_SCORE_NAMES
        }
        sums = {name: {} for name in COMPONENT_SCORE_NAMES} if compute_metrics else None
        rows = 0
        for block in iter_component_blocks(data, split, args, device):
            neg_mask = block.valid[:, 1:]
            for name in COMPONENT_SCORE_NAMES:
                score = block.scores[name]
                pos = score[:, :1]
                neg = score[:, 1:]
                writers[name].write_batch(pos, neg, neg_mask)
                if compute_metrics:
                    add_metric_sums(sums[name], compute_ranking_metric_sums(pos, neg, neg_mask))
            rows += int(block.valid.shape[0])
        split_metrics = {}
        for name in COMPONENT_SCORE_NAMES:
            writers[name].close()
            if compute_metrics:
                m = finalize_metric_sums(sums[name])
                m["num_queries"] = int(sums[name].get("count", 0))
                split_metrics[name] = m
        metrics["splits"][split] = {"rows": int(rows), "metrics": split_metrics}
        if compute_metrics:
            print(
                f"[ComponentScores] struct={struct_id} split={split} rows={rows} "
                f"structure_raw {format_metrics(split_metrics['structure_raw'])}",
                flush=True,
            )
        else:
            print(f"[ComponentScores] struct={struct_id} split={split} rows={rows} metrics=skipped", flush=True)
    with open(osp.join(root, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return root, metrics


def predict_lgbm(model, X):
    pred = model.predict(X).astype(np.float32, copy=False)
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("LGBM produced non-finite predictions")
    return pred


def evaluate_structure_model(data, split, args, feature_builder, model, device):
    sums = {}
    for block in iter_structure_blocks(data, split, args, feature_builder, device):
        X_parts = []
        slices = []
        cursor = 0
        for row in range(block.valid.shape[0]):
            cols = np.flatnonzero(block.valid[row])
            X_parts.append(block.cube[row, cols, :])
            slices.append((cursor, cursor + len(cols)))
            cursor += len(cols)
        if not X_parts:
            continue
        pred = predict_lgbm(model, np.vstack(X_parts).astype(np.float32, copy=False))
        loose = []
        strict = []
        avg = []
        for start, end in slices:
            scores = pred[start:end]
            pos_score = scores[0]
            neg = scores[1:]
            l_rank = 1 + int(np.sum(neg > pos_score))
            s_rank = 1 + int(np.sum(neg >= pos_score))
            loose.append(l_rank)
            strict.append(s_rank)
            avg.append((l_rank + s_rank) * 0.5)
        add_rank_sums(
            sums,
            np.asarray(loose, dtype=np.int64),
            np.asarray(strict, dtype=np.int64),
            np.asarray(avg, dtype=np.float64),
        )
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


class HybridFeatureBuilder:
    def __init__(self, num_rels):
        self.num_rels = int(num_rels)
        self.feature_names = []
        self._init_names()

    def _add(self, name):
        self.feature_names.append(name)

    def _init_names(self):
        for prefix in ("struct", "time", "base"):
            self._add(f"{prefix}_score")
            self._add(f"{prefix}_z")
            self._add(f"{prefix}_minmax")
            self._add(f"{prefix}_rank_log")
            self._add(f"{prefix}_rank_recip")
        for name in (
            "struct_minus_time",
            "abs_struct_minus_time",
            "struct_times_time",
            "score_mean",
            "score_max",
            "score_min",
            "rank_min",
            "rank_gap",
            "both_top10",
            "either_top10",
            "both_top50",
            "either_top50",
            "relation_is_inverse",
            "candidate_is_source",
        ):
            self._add(name)

    def make(self, struct_scores, time_scores, base_scores, valid, batch_data, cand_ids):
        ranks = {
            "struct": dense_rank(struct_scores, valid),
            "time": dense_rank(time_scores, valid),
            "base": dense_rank(base_scores, valid),
        }
        score_map = {"struct": struct_scores, "time": time_scores, "base": base_scores}
        features = []
        for prefix in ("struct", "time", "base"):
            score = np.where(valid, score_map[prefix], 0.0).astype(np.float32, copy=False)
            rank = ranks[prefix].astype(np.float32, copy=False)
            features.extend(
                [
                    score,
                    zscore_by_query(score, valid),
                    minmax_by_query(score, valid),
                    np.log1p(rank).astype(np.float32, copy=False),
                    (1.0 / np.maximum(rank, 1.0)).astype(np.float32, copy=False),
                ]
            )
        struct_rank = ranks["struct"].astype(np.float32, copy=False)
        time_rank = ranks["time"].astype(np.float32, copy=False)
        features.extend(
            [
                (struct_scores - time_scores).astype(np.float32, copy=False),
                np.abs(struct_scores - time_scores).astype(np.float32, copy=False),
                (struct_scores * time_scores).astype(np.float32, copy=False),
                ((struct_scores + time_scores) * 0.5).astype(np.float32, copy=False),
                np.maximum(struct_scores, time_scores).astype(np.float32, copy=False),
                np.minimum(struct_scores, time_scores).astype(np.float32, copy=False),
                np.minimum(struct_rank, time_rank).astype(np.float32, copy=False),
                np.abs(struct_rank - time_rank).astype(np.float32, copy=False),
                ((struct_rank <= 10) & (time_rank <= 10)).astype(np.float32),
                ((struct_rank <= 10) | (time_rank <= 10)).astype(np.float32),
                ((struct_rank <= 50) & (time_rank <= 50)).astype(np.float32),
                ((struct_rank <= 50) | (time_rank <= 50)).astype(np.float32),
            ]
        )
        rels = batch_data[:, 1].astype(np.int64, copy=False)
        sources = batch_data[:, 0].astype(np.int64, copy=False)
        features.append((rels.reshape(-1, 1) >= self.num_rels // 2).repeat(valid.shape[1], axis=1).astype(np.float32))
        features.append((cand_ids == sources.reshape(-1, 1)).astype(np.float32, copy=False))
        cube = np.stack(features, axis=2).astype(np.float32, copy=False)
        bad = int(np.size(cube) - np.sum(np.isfinite(cube)))
        if bad:
            raise RuntimeError(f"hybrid feature cube has {bad} non-finite values")
        return cube


class RescueHybridFeatureBuilder:
    def __init__(self, num_rels):
        self.num_rels = int(num_rels)
        self.feature_names = []
        self._init_names()

    def _add(self, name):
        self.feature_names.append(name)

    def _init_names(self):
        for prefix in ("structure_raw", "time", "dsh", "dmh", "direct", "shared", "b", "base", "time_structure_base"):
            self._add(f"{prefix}_score")
            self._add(f"{prefix}_z")
            self._add(f"{prefix}_minmax")
            self._add(f"{prefix}_rank_log")
            self._add(f"{prefix}_rank_recip")
            self._add(f"{prefix}_top10")
            self._add(f"{prefix}_top50")
            self._add(f"{prefix}_top100")
        for name in (
            "structure_minus_time",
            "direct_minus_time",
            "dsh_minus_time",
            "dmh_minus_time",
            "shared_minus_time",
            "b_minus_time",
            "abs_structure_minus_time",
            "structure_times_time",
            "direct_times_time",
            "component_mean",
            "component_max",
            "component_min",
            "component_std",
            "rank_min_structure_time",
            "rank_gap_structure_time",
            "rank_gap_direct_time",
            "structure_and_time_top10",
            "structure_or_time_top10",
            "structure_and_time_top50",
            "structure_or_time_top50",
            "b_count_log1p",
            "b_seen",
            "tail_count_log1p",
            "incident_count_log1p",
            "rel_tail_count_log1p",
            "source_count_log1p",
            "rel_count_log1p",
            "source_rel_log1p",
            "exact_sro_log1p",
            "source_tail_log1p",
            "relation_is_inverse",
            "candidate_is_source",
        ):
            self._add(name)

    def make(self, scores, time_scores, valid, batch_data, cand_ids, b_counts=None, history_values=None):
        score_map = {
            "structure_raw": scores["structure_raw"],
            "time": time_scores,
            "dsh": scores["dsh"],
            "dmh": scores["dmh"],
            "direct": scores["direct"],
            "shared": scores["shared"],
            "b": scores["b"],
            "base": scores["base"],
        }
        score_map["time_structure_base"] = (
            (minmax_by_query(score_map["structure_raw"], valid) + minmax_by_query(time_scores, valid)) * 0.5
        ).astype(np.float32, copy=False)
        ranks = {name: dense_rank(value, valid).astype(np.float32, copy=False) for name, value in score_map.items()}

        features = []
        for prefix in ("structure_raw", "time", "dsh", "dmh", "direct", "shared", "b", "base", "time_structure_base"):
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
        b = score_map["b"]
        stack = np.stack([structure, time_score, dsh, dmh, direct, shared, b, score_map["base"]], axis=2)
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
                (b - time_score).astype(np.float32, copy=False),
                np.abs(structure - time_score).astype(np.float32, copy=False),
                (structure * time_score).astype(np.float32, copy=False),
                (direct * time_score).astype(np.float32, copy=False),
                np.mean(stack, axis=2).astype(np.float32, copy=False),
                np.max(stack, axis=2).astype(np.float32, copy=False),
                np.min(stack, axis=2).astype(np.float32, copy=False),
                np.std(stack, axis=2).astype(np.float32, copy=False),
                np.minimum(sr, tr).astype(np.float32, copy=False),
                np.abs(sr - tr).astype(np.float32, copy=False),
                np.abs(dr - tr).astype(np.float32, copy=False),
                ((sr <= 10) & (tr <= 10) & valid).astype(np.float32),
                (((sr <= 10) | (tr <= 10)) & valid).astype(np.float32),
                ((sr <= 50) & (tr <= 50) & valid).astype(np.float32),
                (((sr <= 50) | (tr <= 50)) & valid).astype(np.float32),
            ]
        )
        if b_counts is None:
            b_counts = np.zeros_like(structure, dtype=np.float32)
        features.append(np.log1p(np.maximum(b_counts, 0.0)).astype(np.float32, copy=False))
        features.append((b_counts > 0).astype(np.float32, copy=False))
        if history_values is None:
            history_values = [np.zeros_like(structure, dtype=np.float32) for _ in range(8)]
        for arr in history_values:
            features.append(np.log1p(np.maximum(arr, 0.0)).astype(np.float32, copy=False))
        rels = batch_data[:, 1].astype(np.int64, copy=False)
        sources = batch_data[:, 0].astype(np.int64, copy=False)
        features.append((rels.reshape(-1, 1) >= self.num_rels // 2).repeat(valid.shape[1], axis=1).astype(np.float32))
        features.append((cand_ids == sources.reshape(-1, 1)).astype(np.float32, copy=False))
        cube = np.stack(features, axis=2).astype(np.float32, copy=False)
        bad = int(np.size(cube) - np.sum(np.isfinite(cube)))
        if bad:
            raise RuntimeError(f"rescue hybrid feature cube has {bad} non-finite values")
        return cube


def base_hybrid_scores(struct_scores, time_scores, valid):
    return ((minmax_by_query(struct_scores, valid) + minmax_by_query(time_scores, valid)) * 0.5).astype(
        np.float32, copy=False
    )


def iter_hybrid_blocks(data, split, args, structure_feature_builder, hybrid_feature_builder, structure_model, time_dir, device):
    time_store = ScoreStore(time_dir, split)
    row_offset = 0
    for block in iter_structure_blocks(data, split, args, structure_feature_builder, device):
        width = block.valid.shape[1] - 1
        end = row_offset + block.valid.shape[0]
        time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
        time_valid = np.concatenate((np.ones((len(time_pos), 1), dtype=bool), time_mask), axis=1)
        valid = block.valid & time_valid
        X_parts = []
        slices = []
        cursor = 0
        for row in range(valid.shape[0]):
            cols = np.flatnonzero(valid[row])
            X_parts.append(block.cube[row, cols, :])
            slices.append((cursor, cursor + len(cols), cols))
            cursor += len(cols)
        struct_scores = np.zeros(valid.shape, dtype=np.float32)
        if X_parts:
            pred = predict_lgbm(structure_model, np.vstack(X_parts).astype(np.float32, copy=False))
            for row, (start, end_s, cols) in enumerate(slices):
                struct_scores[row, cols] = pred[start:end_s]
        time_scores = np.concatenate((time_pos, time_neg), axis=1).astype(np.float32, copy=False)
        time_scores = np.where(valid, time_scores, 0.0).astype(np.float32, copy=False)
        base_scores = base_hybrid_scores(struct_scores, time_scores, valid)
        features = hybrid_feature_builder.make(
            struct_scores,
            time_scores,
            base_scores,
            valid,
            block.batch_data,
            block.cand_ids,
        )
        yield SimpleNamespace(
            batch_data=block.batch_data,
            valid=valid,
            struct_scores=struct_scores,
            time_scores=time_scores,
            base_scores=base_scores,
            cand_ids=block.cand_ids,
            features=features,
            t_norm=block.t_norm,
            t_orig=block.t_orig,
        )
        row_offset = end
    if row_offset != time_store.num_rows:
        raise RuntimeError(f"time row count mismatch for split={split}: stream={row_offset}, store={time_store.num_rows}")


def selected_hybrid(struct_scores, time_scores, base_scores, valid, topk):
    if int(topk) < 0:
        selected = valid.copy()
    else:
        selected = (
            ((dense_rank(struct_scores, valid) <= int(topk) + 1) & valid)
            | ((dense_rank(time_scores, valid) <= int(topk) + 1) & valid)
            | ((dense_rank(base_scores, valid) <= int(topk) + 1) & valid)
        )
    selected[:, 0] = True
    return selected & valid


def build_hybrid_matrix(data, split, args, structure_feature_builder, hybrid_feature_builder, structure_model, time_dir, device, topk):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0
    for block in iter_hybrid_blocks(
        data, split, args, structure_feature_builder, hybrid_feature_builder, structure_model, time_dir, device
    ):
        selected = selected_hybrid(block.struct_scores, block.time_scores, block.base_scores, block.valid, topk)
        for row in range(selected.shape[0]):
            cols = np.flatnonzero(selected[row])
            if len(cols) <= 1:
                continue
            pos = np.flatnonzero(cols == 0)
            if len(pos) != 1:
                raise RuntimeError(f"{split} hybrid matrix row missing positive")
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[int(pos[0])] = 1.0
            X_parts.append(block.features[row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)
    if not X_parts:
        raise RuntimeError(f"no hybrid rows built for split={split}")
    positives = int(np.sum(np.concatenate(y_parts) > 0.0))
    if positives != queries:
        raise RuntimeError(f"hybrid matrix expected one positive per query, got positives={positives}, queries={queries}")
    nonfinite = int(sum(np.size(x) - np.sum(np.isfinite(x)) for x in X_parts))
    if nonfinite:
        raise RuntimeError(f"hybrid matrix has {nonfinite} non-finite feature values")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {"split": split, "queries": int(queries), "rows": int(rows), "topk": int(topk)},
    )


def evaluate_hybrid_model(
    data,
    split,
    args,
    structure_feature_builder,
    hybrid_feature_builder,
    structure_model,
    hybrid_model,
    time_dir,
    device,
    save_top10_path=None,
):
    sums = {}
    top10_f = None
    if save_top10_path:
        ensure_dir(osp.dirname(save_top10_path))
        top10_f = open(save_top10_path, "w", encoding="utf-8")
    query_idx = 0
    try:
        for block in iter_hybrid_blocks(
            data, split, args, structure_feature_builder, hybrid_feature_builder, structure_model, time_dir, device
        ):
            X_parts = []
            slices = []
            cursor = 0
            for row in range(block.valid.shape[0]):
                cols = np.flatnonzero(block.valid[row])
                X_parts.append(block.features[row, cols, :])
                slices.append((cursor, cursor + len(cols), row, cols))
                cursor += len(cols)
            if not X_parts:
                continue
            pred = predict_lgbm(hybrid_model, np.vstack(X_parts).astype(np.float32, copy=False))
            loose = []
            strict = []
            avg = []
            for start, end, row, cols in slices:
                scores = pred[start:end]
                pos_score = scores[0]
                neg = scores[1:]
                l_rank = 1 + int(np.sum(neg > pos_score))
                s_rank = 1 + int(np.sum(neg >= pos_score))
                loose.append(l_rank)
                strict.append(s_rank)
                avg.append((l_rank + s_rank) * 0.5)
                if top10_f is not None:
                    top = np.argsort(-scores, kind="stable")[:10]
                    event = block.batch_data[row]
                    payload = {
                        "query_index": int(query_idx),
                        "s": int(event[0]),
                        "r": int(event[1]),
                        "o": int(event[2]),
                        "t_norm": float(block.t_norm),
                        "t_orig": int(block.t_orig),
                        "strict_rank": int(s_rank),
                        "loose_rank": int(l_rank),
                        "top10": [
                            {
                                "rank": int(k + 1),
                                "candidate_id": int(block.cand_ids[row, cols[int(col)]]),
                                "is_positive": bool(cols[int(col)] == 0),
                                "hybrid_score": float(scores[int(col)]),
                                "structure_score": float(block.struct_scores[row, cols[int(col)]]),
                                "time_score": float(block.time_scores[row, cols[int(col)]]),
                            }
                            for k, col in enumerate(top)
                        ],
                    }
                    top10_f.write(json.dumps(payload, separators=(",", ":")) + "\n")
                query_idx += 1
            add_rank_sums(
                sums,
                np.asarray(loose, dtype=np.int64),
                np.asarray(strict, dtype=np.int64),
                np.asarray(avg, dtype=np.float64),
            )
    finally:
        if top10_f is not None:
            top10_f.close()
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


def iter_rescue_hybrid_blocks(data, split, args, rescue_feature_builder, time_dir, device):
    time_store = ScoreStore(time_dir, split)
    row_offset = 0
    for block in iter_component_blocks(data, split, args, device):
        width = block.valid.shape[1] - 1
        end = row_offset + block.valid.shape[0]
        time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
        time_valid = np.concatenate((np.ones((len(time_pos), 1), dtype=bool), time_mask), axis=1)
        valid = block.valid & time_valid
        time_scores = np.concatenate((time_pos, time_neg), axis=1).astype(np.float32, copy=False)
        time_scores = np.where(valid, time_scores, 0.0).astype(np.float32, copy=False)
        features = rescue_feature_builder.make(
            block.scores,
            time_scores,
            valid,
            block.batch_data,
            block.cand_ids,
            b_counts=block.b_counts,
            history_values=block.history_values,
        )
        yield SimpleNamespace(
            batch_data=block.batch_data,
            valid=valid,
            scores=block.scores,
            time_scores=time_scores,
            cand_ids=block.cand_ids,
            features=features,
            t_norm=block.t_norm,
            t_orig=block.t_orig,
        )
        row_offset = end
    if row_offset != time_store.num_rows:
        raise RuntimeError(f"time row count mismatch for split={split}: stream={row_offset}, store={time_store.num_rows}")


def iter_saved_rescue_hybrid_blocks(data, split, args, rescue_feature_builder, component_root, time_dir):
    time_store = ScoreStore(time_dir, split)
    component_stores = {
        name: ScoreStore(osp.join(component_root, split, name), split)
        for name in COMPONENT_SCORE_NAMES
    }
    expected_rows = next(iter(component_stores.values())).num_rows
    for name, store in component_stores.items():
        if store.num_rows != expected_rows:
            raise RuntimeError(f"component row mismatch for {split}/{name}: {store.num_rows} != {expected_rows}")
    if time_store.num_rows != expected_rows:
        raise RuntimeError(f"time/component row mismatch for {split}: time={time_store.num_rows}, component={expected_rows}")

    timeline, history = init_timeline_history_state(data, split, args)
    neg_sampler = data["negative_sampler"]
    row_offset = 0
    for events, t_norm, t_orig in split_snapshots(data, split):
        for batch_data, neg_arr, neg_mask in collect_eval_batch(events, t_orig, neg_sampler, split, args.query_batch_size):
            width = int(neg_arr.shape[1])
            end = row_offset + len(batch_data)
            scores = {}
            valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), neg_mask[:, :width]), axis=1)
            for name, store in component_stores.items():
                pos, neg, mask = store.get_block(row_offset, end, width)
                valid[:, 1:] &= mask
                scores[name] = np.concatenate((pos, neg), axis=1).astype(np.float32, copy=False)
            time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
            valid[:, 1:] &= time_mask
            time_scores = np.concatenate((time_pos, time_neg), axis=1).astype(np.float32, copy=False)
            time_scores = np.where(valid, time_scores, 0.0).astype(np.float32, copy=False)

            rels, pos_counts, neg_counts = timeline.count_batch(batch_data, neg_arr, width)
            cand_ids = np.concatenate((batch_data[:, 2:3], neg_arr[:, :width]), axis=1)
            cand_ids = np.where(valid, cand_ids, -1).astype(np.int64, copy=False)
            b_counts = np.concatenate((pos_counts.reshape(-1, 1), neg_counts), axis=1).astype(np.float32, copy=False)
            history_values = history.features(
                batch_data[:, 0].astype(np.int64, copy=False),
                batch_data[:, 1].astype(np.int64, copy=False),
                cand_ids,
                b_counts,
            )
            features = rescue_feature_builder.make(
                scores,
                time_scores,
                valid,
                batch_data,
                cand_ids,
                b_counts=b_counts,
                history_values=history_values,
            )
            yield SimpleNamespace(
                batch_data=batch_data,
                valid=valid,
                scores=scores,
                time_scores=time_scores,
                cand_ids=cand_ids,
                features=features,
                t_norm=t_norm,
                t_orig=t_orig,
            )
            row_offset = end

        update_events = events_for_update(events, data, args, is_train=(split == "train"))
        timeline.update(update_events)
        history.update(update_events)

    if row_offset != expected_rows:
        raise RuntimeError(f"component row count mismatch for split={split}: stream={row_offset}, stores={expected_rows}")


def iter_rescue_blocks(data, split, args, rescue_feature_builder, time_dir, device, component_root=None):
    if component_root:
        yield from iter_saved_rescue_hybrid_blocks(data, split, args, rescue_feature_builder, component_root, time_dir)
    else:
        yield from iter_rescue_hybrid_blocks(data, split, args, rescue_feature_builder, time_dir, device)


def rescue_topk_mask(structure_scores, valid, topk):
    ranks = dense_rank(structure_scores, valid)
    return (ranks <= int(topk)) & valid, ranks


def build_rescue_hybrid_matrix(
    data,
    split,
    args,
    rescue_feature_builder,
    time_dir,
    device,
    topk,
    component_root=None,
    min_pos_rank=1,
    max_pos_rank=100,
    include_top10=True,
    max_queries=0,
    query_stride=1,
):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0
    skipped_pos_after_topk = 0
    preserve_queries = 0
    rescue_queries = 0
    eligible_queries = 0
    sampled_out_queries = 0
    max_queries = int(max_queries or 0)
    query_stride = max(1, int(query_stride or 1))
    for block in iter_rescue_blocks(data, split, args, rescue_feature_builder, time_dir, device, component_root):
        selected, ranks = rescue_topk_mask(block.scores["structure_raw"], block.valid, topk)
        pos_ranks = ranks[:, 0]
        for row in range(selected.shape[0]):
            if max_queries > 0 and queries >= max_queries:
                break
            pos_rank = int(pos_ranks[row])
            if pos_rank > int(max_pos_rank):
                skipped_pos_after_topk += 1
                continue
            if pos_rank < int(min_pos_rank) and not include_top10:
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
            X_parts.append(block.features[row, cols, :])
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
        raise RuntimeError(f"no rescue hybrid rows built for split={split}")
    positives = int(np.sum(np.concatenate(y_parts) > 0.0))
    if positives != queries:
        raise RuntimeError(f"rescue hybrid matrix expected one positive per query, got positives={positives}, queries={queries}")
    nonfinite = int(sum(np.size(x) - np.sum(np.isfinite(x)) for x in X_parts))
    if nonfinite:
        raise RuntimeError(f"rescue hybrid matrix has {nonfinite} non-finite feature values")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {
            "split": split,
            "queries": int(queries),
            "rows": int(rows),
            "topk": int(topk),
            "min_pos_rank": int(min_pos_rank),
            "max_pos_rank": int(max_pos_rank),
            "include_top10": bool(include_top10),
            "preserve_queries": int(preserve_queries),
            "rescue_queries": int(rescue_queries),
            "skipped_pos_after_topk": int(skipped_pos_after_topk),
            "eligible_queries": int(eligible_queries),
            "sampled_out_queries": int(sampled_out_queries),
            "max_queries": int(max_queries),
            "query_stride": int(query_stride),
        },
    )


def _score_rescue_selected(raw_scores, selected, pred):
    final = raw_scores.astype(np.float32, copy=True)
    if not np.any(selected):
        return final
    outside = (~selected) & np.isfinite(final)
    floor = float(np.max(final[outside])) if np.any(outside) else 0.0
    pred = np.asarray(pred, dtype=np.float32)
    if pred.size == 0:
        return final
    selected_cols = np.flatnonzero(selected)
    raw_sel = raw_scores[selected_cols].astype(np.float32, copy=False)
    order = np.lexsort((selected_cols, -raw_sel, -pred))
    ordered_cols = selected_cols[order]
    n = int(len(ordered_cols))
    final[ordered_cols] = floor + 1.0 + ((n - np.arange(n, dtype=np.float32)) / max(n, 1))
    return final


def evaluate_rescue_hybrid_model(
    data,
    split,
    args,
    rescue_feature_builder,
    hybrid_model,
    time_dir,
    device,
    topk,
    component_root=None,
    save_top10_path=None,
):
    sums = {}
    rescue_stats = {
        "pos_in_top10_before": 0,
        "pos_11_100_before": 0,
        "pos_after_top100_before": 0,
        "pos_in_top10_after": 0,
        "rescued_11_100_to_top10": 0,
        "dropped_top10": 0,
    }
    top10_f = None
    if save_top10_path:
        ensure_dir(osp.dirname(save_top10_path))
        top10_f = open(save_top10_path, "w", encoding="utf-8")
    query_idx = 0
    try:
        for block in iter_rescue_blocks(data, split, args, rescue_feature_builder, time_dir, device, component_root):
            selected, ranks = rescue_topk_mask(block.scores["structure_raw"], block.valid, topk)
            loose = []
            strict = []
            avg = []
            for row in range(block.valid.shape[0]):
                valid_cols = np.flatnonzero(block.valid[row])
                sel_cols = np.flatnonzero(selected[row])
                raw_scores = np.where(block.valid[row], block.scores["structure_raw"][row], -np.inf).astype(np.float32)
                before_rank = int(ranks[row, 0])
                if before_rank <= 10:
                    rescue_stats["pos_in_top10_before"] += 1
                elif before_rank <= int(topk):
                    rescue_stats["pos_11_100_before"] += 1
                else:
                    rescue_stats["pos_after_top100_before"] += 1
                if len(sel_cols):
                    X = block.features[row, sel_cols, :].astype(np.float32, copy=False)
                    pred = predict_lgbm(hybrid_model, X)
                    final_scores = _score_rescue_selected(raw_scores, selected[row], pred)
                else:
                    final_scores = raw_scores
                    pred = np.empty(0, dtype=np.float32)
                pos_score = final_scores[0]
                neg_scores = final_scores[1:]
                neg_valid = block.valid[row, 1:]
                l_rank = 1 + int(np.sum((neg_scores > pos_score) & neg_valid))
                s_rank = 1 + int(np.sum((neg_scores >= pos_score) & neg_valid))
                loose.append(l_rank)
                strict.append(s_rank)
                avg.append((l_rank + s_rank) * 0.5)
                if s_rank <= 10:
                    rescue_stats["pos_in_top10_after"] += 1
                    if 10 < before_rank <= int(topk):
                        rescue_stats["rescued_11_100_to_top10"] += 1
                elif before_rank <= 10:
                    rescue_stats["dropped_top10"] += 1
                if top10_f is not None:
                    top = valid_cols[np.argsort(-final_scores[valid_cols], kind="stable")[:10]]
                    event = block.batch_data[row]
                    pred_map = {int(col): float(score) for col, score in zip(sel_cols, pred)}
                    payload = {
                        "query_index": int(query_idx),
                        "s": int(event[0]),
                        "r": int(event[1]),
                        "o": int(event[2]),
                        "t_norm": float(block.t_norm),
                        "t_orig": int(block.t_orig),
                        "structure_rank_before": int(before_rank),
                        "strict_rank": int(s_rank),
                        "loose_rank": int(l_rank),
                        "top10": [
                            {
                                "rank": int(k + 1),
                                "candidate_id": int(block.cand_ids[row, int(col)]),
                                "is_positive": bool(int(col) == 0),
                                "hybrid_score": float(final_scores[int(col)]),
                                "lgbm_score": pred_map.get(int(col)),
                                "structure_score": float(block.scores["structure_raw"][row, int(col)]),
                                "time_score": float(block.time_scores[row, int(col)]),
                            }
                            for k, col in enumerate(top)
                        ],
                    }
                    top10_f.write(json.dumps(payload, separators=(",", ":")) + "\n")
                query_idx += 1
            add_rank_sums(
                sums,
                np.asarray(loose, dtype=np.int64),
                np.asarray(strict, dtype=np.int64),
                np.asarray(avg, dtype=np.float64),
            )
    finally:
        if top10_f is not None:
            top10_f.close()
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    metrics["rescue_stats"] = rescue_stats
    return metrics


def evaluate_score_store(out_dir, data, split, args):
    store = ScoreStore(out_dir, split)
    sums = {}
    row_offset = 0
    for events, _, t_orig in split_snapshots(data, split):
        for batch_data, neg_arr, neg_mask in collect_eval_batch(
            events, t_orig, data["negative_sampler"], split, args.query_batch_size
        ):
            width = int(neg_arr.shape[1])
            end = row_offset + len(batch_data)
            pos, neg, mask = store.get_block(row_offset, end, width)
            valid_mask = neg_mask[:, :width] & mask
            add_metric_sums(sums, compute_ranking_metric_sums(pos, neg, valid_mask))
            row_offset = end
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


def fit_lgbm_ranker(X, y, group, feature_names, categorical_indices, args, params):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lightgbm is required") from exc
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        min_split_gain=float(params["min_split_gain"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        random_state=int(args.seed),
        n_jobs=int(args.num_threads),
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    fit_kwargs = {"group": group.tolist(), "feature_name": feature_names}
    if categorical_indices:
        fit_kwargs["categorical_feature"] = categorical_indices
    model.fit(X, y, **fit_kwargs)
    return model


def save_lgbm_model(model, path):
    ensure_dir(osp.dirname(path))
    booster = getattr(model, "booster_", model)
    booster.save_model(path)
