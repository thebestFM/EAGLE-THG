import gc
import hashlib
import json
import os
import os.path as osp
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from tqdm import tqdm

from utils import (
    HIT_KS,
    SUPPORTED_DATASETS,
    TGB_DATASETS,
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    finalize_metric_sums,
    inverse_aug,
    is_run_complete,
    load_config,
    load_datasets,
    load_metrics,
    ranking_metric_key,
    save_config,
    save_metrics,
    set_random_seed,
)


EPS = 1e-12
LGBM_FIT_PROTOCOL = "stream_val_metric_v1"


@dataclass(frozen=True)
class BConfig:
    mode: str
    binary_unseen: float
    continuous_alpha: float


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
        if end > self.num_rows:
            raise ValueError(f"score row overflow: {self.out_dir}/{self.mode}, end={end}, rows={self.num_rows}")
        width = min(int(width), self.max_negs)
        neg = self.neg[start:end, :width]
        if neg.dtype != np.float32:
            neg = neg.astype(np.float32)
        neg = neg.toarray().astype(np.float32, copy=False)
        lens = np.minimum(self.valid_lens[start:end], width)
        mask = np.arange(width)[None, :] < lens[:, None]
        return (
            self.pos[start:end].astype(np.float32, copy=False),
            np.where(mask, neg, 0.0).astype(np.float32, copy=False),
            mask,
        )


class BTimeline:
    def __init__(self, num_rels, num_nodes):
        self.num_rels = int(num_rels)
        self.num_nodes = int(num_nodes)
        self.counts_ro = csr_matrix((self.num_rels, self.num_nodes), dtype=np.float32)
        self.counts_r = np.zeros(self.num_rels, dtype=np.float32)

    def update(self, events):
        rels, objs, vals = event_delta(events, self.num_nodes)
        if len(vals) == 0:
            return
        delta = csr_matrix((vals, (rels, objs)), shape=self.counts_ro.shape, dtype=np.float32)
        self.counts_ro = (self.counts_ro + delta).tocsr()
        self.counts_r += np.bincount(rels, weights=vals, minlength=self.num_rels).astype(np.float32)

    def count_batch(self, batch_data, neg_arr, width):
        rels = batch_data[:, 1].astype(np.int64)
        pos_obj = batch_data[:, 2].astype(np.int64)
        neg_obj = np.where(neg_arr[:, :width] < 0, 0, neg_arr[:, :width]).astype(np.int64)

        pos_counts = np.zeros(len(batch_data), dtype=np.float32)
        neg_counts = np.zeros((len(batch_data), width), dtype=np.float32)
        for rel in np.unique(rels):
            rows = np.flatnonzero(rels == rel)
            dense = self.counts_ro.getrow(int(rel)).toarray().ravel().astype(np.float32, copy=False)
            pos_counts[rows] = dense[pos_obj[rows]]
            neg_counts[rows] = dense[neg_obj[rows]]
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
        self.source_rel = defaultdict(float)
        self.sr_tail = defaultdict(Counter)
        self.source_tail = defaultdict(Counter)

    def update(self, events):
        if len(events) == 0:
            return
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
            self.source_rel[(s, r)] += 1.0
            self.sr_tail[(s, r)][o] += 1
            self.source_tail[s][o] += 1

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
            sr_counter = self.sr_tail.get((int(sources[i]), int(rels[i])))
            st_counter = self.source_tail.get(int(sources[i]))
            if sr_counter:
                exact_sro[i, :] = [sr_counter.get(int(o), 0) for o in cand_ids[i]]
            if st_counter:
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


class FeatureBuilder:
    def __init__(self, num_rels):
        self.num_rels = int(num_rels)
        self.feature_names = []
        self.categorical_names = []
        self._init_names()

    def _add(self, name, categorical=False):
        self.feature_names.append(name)
        if categorical:
            self.categorical_names.append(name)

    def _init_names(self):
        for prefix in ("base", "a", "b", "c"):
            self._add(f"{prefix}_score")
            self._add(f"{prefix}_log1p")
            self._add(f"{prefix}_max_norm")
            self._add(f"{prefix}_z")
            self._add(f"{prefix}_rank_log")
            self._add(f"{prefix}_rank_recip")

        for name in (
            "a_minus_b",
            "a_minus_c",
            "b_minus_c",
            "a_times_b",
            "a_times_c",
            "b_times_c",
            "abc_mean",
            "abc_max",
            "abc_std",
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

    @property
    def categorical_indices(self):
        positions = {name: i for i, name in enumerate(self.feature_names)}
        return [positions[name] for name in self.categorical_names]

    def make(self, batch_data, cand_ids, valid, scores, b_counts, history):
        sources = batch_data[:, 0].astype(np.int64)
        rels = batch_data[:, 1].astype(np.int64)
        width = cand_ids.shape[1]

        features = []
        for prefix in ("base", "a", "b", "c"):
            sc = np.where(valid, scores[prefix], 0.0).astype(np.float32, copy=False)
            ranks = dense_rank(sc, valid)
            features.append(sc)
            features.append(np.log1p(np.maximum(sc, 0.0)).astype(np.float32, copy=False))
            features.append(max_norm(sc, valid))
            features.append(zscore_by_query(sc, valid))
            features.append(np.log1p(ranks.astype(np.float32)).astype(np.float32, copy=False))
            features.append((1.0 / np.maximum(ranks, 1)).astype(np.float32, copy=False))

        a = scores["a"]
        b = scores["b"]
        c = scores["c"]
        features.extend(
            [
                (a - b).astype(np.float32, copy=False),
                (a - c).astype(np.float32, copy=False),
                (b - c).astype(np.float32, copy=False),
                (a * b).astype(np.float32, copy=False),
                (a * c).astype(np.float32, copy=False),
                (b * c).astype(np.float32, copy=False),
                ((a + b + c) / 3.0).astype(np.float32, copy=False),
                np.maximum(np.maximum(a, b), c).astype(np.float32, copy=False),
                np.std(np.stack([a, b, c], axis=2), axis=2).astype(np.float32, copy=False),
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

        return np.stack(features, axis=2).astype(np.float32, copy=False)


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


def safe_token(text):
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(text))


OUTPUT_ARG_KEYS = (
    "dataset",
    "seed",
    "ns_q",
    "ns_seed",
    "train_predict_ratio",
    "metric",
    "component_metric",
    "a_prefix",
    "c_prefix",
    "top_a",
    "top_b",
    "top_c",
    "close_update_backward",
    "train_topk",
    "b_modes",
    "binary_unseen_grid",
    "continuous_alpha_grid",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "reg_lambda",
    "reg_alpha",
    "min_split_gain",
    "subsample",
    "colsample_bytree",
    "lgbm_n_trials",
    "lgbm_early_stopping_rounds",
)


def compact_value(value):
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def stable_hash(payload, length=12):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[: int(length)]


def output_arg_dict(args):
    return {key: getattr(args, key) for key in OUTPUT_ARG_KEYS if hasattr(args, key)}


def make_out_dir(args):
    cfg_hash = stable_hash(output_arg_dict(args), length=12)
    b_hash = stable_hash(
        {
            "modes": args.b_modes,
            "close_update_backward": bool(getattr(args, "close_update_backward", False)) if args.dataset in TGB_DATASETS else False,
            "binary_unseen_grid": args.binary_unseen_grid,
            "continuous_alpha_grid": args.continuous_alpha_grid,
        },
        length=10,
    )
    lgbm_hash = stable_hash(
        {
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
            "lgbm_n_trials": args.lgbm_n_trials,
            "lgbm_early_stopping_rounds": args.lgbm_early_stopping_rounds,
            "fit_protocol": LGBM_FIT_PROTOCOL,
        },
        length=8,
    )
    parts = [
        args.out_prefix,
        args.dataset,
        f"seed{args.seed}",
        f"tr{compact_value(args.train_predict_ratio)}_nq{args.ns_q}_ns{args.ns_seed}",
        f"m-{safe_token(args.metric)}_cm-{safe_token(args.component_metric)}",
        f"top{args.top_a}-{args.top_b}-{args.top_c}_traink{args.train_topk}_evalfull",
        f"b-{b_hash}",
        f"lgbm-{lgbm_hash}",
        f"cfg-{cfg_hash}",
    ]
    return osp.join(*parts)


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise OSError(f"cannot create output directory ({len(path)} chars): {path}") from exc


def save_lgbm_model(model, path):
    booster = getattr(model, "booster_", model)
    booster.save_model(path)


def load_lgbm_model(path):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("combine_lgbm.py requires lightgbm to load cached combiner models") from exc
    return lgb.Booster(model_file=path)


def make_combo_key(args, a_run, c_run, b_cfg):
    payload = {
        "args": output_arg_dict(args),
        "fit_protocol": LGBM_FIT_PROTOCOL,
        "a_dir": osp.basename(a_run["dir"]),
        "a_config": short_config(a_run["config"]),
        "c_dir": osp.basename(c_run["dir"]),
        "c_config": short_config(c_run["config"]),
        "b_config": asdict(b_cfg),
    }
    return stable_hash(payload, length=16)


def combo_cache_dir(out_dir, combo_key):
    return osp.join(out_dir, "combo_cache", f"combo-{combo_key}")


def load_combo_cache(out_dir, combo_key):
    cache_dir = combo_cache_dir(out_dir, combo_key)
    record_path = osp.join(cache_dir, "record.json")
    model_path = osp.join(cache_dir, "model.txt")
    if not osp.isfile(record_path) or not osp.isfile(model_path):
        return None
    try:
        with open(record_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[combine] skip combo cache {cache_dir}: {exc}", flush=True)
        return None
    if payload.get("format") != "abc_lgbm_combo_v1" or payload.get("combo_key") != combo_key:
        return None
    return payload["record"], model_path


def save_combo_cache(out_dir, combo_key, record, model):
    cache_dir = combo_cache_dir(out_dir, combo_key)
    ensure_dir(cache_dir)
    model_path = osp.join(cache_dir, "model.txt")
    record_path = osp.join(cache_dir, "record.json")
    save_lgbm_model(model, model_path)
    payload = {
        "format": "abc_lgbm_combo_v1",
        "combo_key": combo_key,
        "record": record,
        "model_path": model_path,
    }
    with open(record_path, "w") as f:
        json.dump(payload, f, indent=2)


def short_config(cfg):
    keys = [
        "seed",
        "ns_q",
        "ns_seed",
        "train_predict_ratio",
        "close_update_backward",
        "a_mode",
        "decay_a",
        "ppr_beta",
        "impl",
        "c_storage",
        "shared_w",
        "ppr_k",
        "ppr_alpha",
        "ppr_beta",
        "gamma",
        "top_share",
        "window_semantic_sim",
        "window_trans",
        "decay_rel_trans",
        "top_k_relation",
        "per_rel_use_mtrans",
    ]
    return {key: cfg[key] for key in keys if key in cfg}


def should_inverse_aug(data, close_update_backward=False):
    return data["is_tgb"] and not bool(close_update_backward)


def metric_value(metrics, metric):
    key = ranking_metric_key(metric, strict=True)
    return float(metrics[key])


def format_metrics(metrics):
    strict_hits = " ".join(f"h@{k}={metrics[f'hit@{k}_strict']:.5f}" for k in HIT_KS)
    loose_hits = " ".join(f"h@{k}={metrics[f'hit@{k}_loose']:.5f}" for k in HIT_KS)
    avg_hits = " ".join(f"h@{k}={metrics[f'hit@{k}_avg']:.5f}" for k in HIT_KS)
    return (
        f"mrr_strict={metrics['mrr_strict']:.5f} "
        f"mrr_loose={metrics['mrr_loose']:.5f} "
        f"mrr_avg={metrics['mrr_avg']:.5f} "
        f"strict[{strict_hits}] loose[{loose_hits}] avg[{avg_hits}]"
    )


def load_single_score_run(out_dir, component, args):
    if not out_dir:
        raise SystemExit(f"--{component.lower()}_dir is required when --mode single")
    required_modes = ("train", "val")
    if getattr(args, "eval_test", True):
        required_modes = required_modes + ("test",)
    if not is_run_complete(out_dir, modes=required_modes):
        raise SystemExit(f"{component} run is incomplete: {out_dir}")
    try:
        cfg = load_config(out_dir)
        metrics = load_metrics(out_dir)
    except Exception as exc:
        raise SystemExit(f"cannot read {component} run {out_dir}: {exc}") from exc
    if str(cfg.get("dataset")) != str(args.dataset):
        raise SystemExit(f"{component} run dataset mismatch: {out_dir}")
    if int(cfg.get("seed", -1)) != int(args.seed):
        raise SystemExit(f"{component} run seed mismatch: {out_dir}")
    if int(cfg.get("ns_q", 10**18)) != int(args.ns_q):
        raise SystemExit(f"{component} run ns_q mismatch: {out_dir}")
    if int(cfg.get("ns_seed", 42)) != int(args.ns_seed):
        raise SystemExit(f"{component} run ns_seed mismatch: {out_dir}")
    if abs(float(cfg.get("train_predict_ratio", -1.0)) - float(args.train_predict_ratio)) > 1e-12:
        raise SystemExit(f"{component} run train_predict_ratio mismatch: {out_dir}")
    if args.dataset in TGB_DATASETS:
        if "close_update_backward" not in cfg:
            raise SystemExit(
                f"{component} run was not produced with close_update_backward-aware code: {out_dir}"
            )
        run_close = bool(cfg.get("close_update_backward", False))
        expected_close = bool(getattr(args, "close_update_backward", False))
        if run_close != expected_close:
            raise SystemExit(
                f"{component} run close_update_backward mismatch: {out_dir} "
                f"(run={int(run_close)} expected={int(expected_close)})"
            )
    return {"dir": out_dir, "config": cfg, "metrics": metrics}


def single_b_config(args):
    return BConfig(
        str(args.b_mode),
        float(args.b_binary_unseen),
        float(args.b_continuous_alpha),
    )


def split_snapshots(data, split):
    if split == "train":
        return data["train_list"][data["train_predict_start_idx"] :]
    return data[f"{split}_list"]


def events_for_update(events, data, close_update_backward=False):
    if should_inverse_aug(data, close_update_backward):
        return inverse_aug(events, data["num_rels_raw"], data["num_rels"])
    return events


def init_stream_state(data, split, close_update_backward=False):
    timeline = BTimeline(data["num_rels"], data["num_nodes"])
    history = CausalHistory(data["num_nodes"], data["num_rels"])

    def apply(snapshots, use_inverse_aug=False):
        for events, _, _ in snapshots:
            update_events = (
                inverse_aug(events, data["num_rels_raw"], data["num_rels"])
                if use_inverse_aug
                else events
            )
            timeline.update(update_events)
            history.update(update_events)

    if split == "train":
        apply(data["train_list"][: data["train_predict_start_idx"]], use_inverse_aug=False)
    elif split == "val":
        apply(data["train_list"], use_inverse_aug=False)
    elif split == "test":
        apply(data["train_list"], use_inverse_aug=False)
        apply(data["val_list"], use_inverse_aug=should_inverse_aug(data, close_update_backward))
    else:
        raise ValueError(f"unknown split: {split}")
    return timeline, history


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
    std = np.where(var <= 1e-12, 1.0, np.sqrt(var))
    return np.where(valid, (scores - mean) / std, 0.0).astype(np.float32, copy=False)


def max_norm(scores, valid):
    max_score = np.max(np.where(valid, scores, 0.0), axis=1, keepdims=True)
    denom = np.where(max_score > 0.0, max_score, 1.0)
    return np.where(valid, scores / denom, 0.0).astype(np.float32, copy=False)


def base_scores(a_scores, b_scores, c_scores, valid):
    return (
        max_norm(a_scores, valid)
        + max_norm(b_scores, valid)
        + max_norm(c_scores, valid)
    ).astype(np.float32, copy=False) / 3.0


def selected_for_training(base_ranks, valid, topk):
    selected = np.zeros_like(valid, dtype=bool)
    selected[:, 0] = True
    selected |= (base_ranks <= int(topk) + 1) & valid
    return selected


def candidate_rank_pair(scores, valid):
    pos = scores[:, :1]
    neg = scores[:, 1:]
    mask = valid[:, 1:]
    loose = 1 + np.sum((neg > pos) & mask, axis=1)
    strict = 1 + np.sum((neg >= pos) & mask, axis=1)
    avg = (loose.astype(np.float64) + strict.astype(np.float64)) * 0.5
    return loose.astype(np.int64), strict.astype(np.int64), avg


def add_rank_sums(total, loose_ranks, strict_ranks, avg_ranks):
    total["count"] = total.get("count", 0) + int(len(loose_ranks))
    for label, ranks in (("loose", loose_ranks), ("strict", strict_ranks), ("avg", avg_ranks)):
        total[f"mrr_{label}"] = total.get(f"mrr_{label}", 0.0) + float(np.sum(1.0 / ranks))
        for k in HIT_KS:
            key = f"hit@{k}_{label}"
            total[key] = total.get(key, 0.0) + float(np.sum(ranks <= int(k)))


def iter_feature_blocks(a_run, c_run, b_cfg, data, split, feature_builder, args):
    snapshots = split_snapshots(data, split)
    a_store = ScoreStore(a_run["dir"], split)
    c_store = ScoreStore(c_run["dir"], split)
    timeline, history = init_stream_state(
        data,
        split,
        close_update_backward=bool(getattr(args, "close_update_backward", False)),
    )
    row_offset = 0

    for events, _, t_orig in snapshots:
        batches = collect_eval_batch(events, t_orig, data["negative_sampler"], split, args.block_size)
        for batch_data, neg_arr, neg_mask in batches:
            width = neg_arr.shape[1]
            if a_store.max_negs < width or c_store.max_negs < width:
                raise ValueError(
                    f"A/C score width mismatch at {split}: sampler={width}, "
                    f"A={a_store.max_negs}, C={c_store.max_negs}"
                )
            end = row_offset + len(batch_data)
            a_pos, a_neg, a_mask = a_store.get_block(row_offset, end, width)
            c_pos, c_neg, c_mask = c_store.get_block(row_offset, end, width)
            rels, pos_counts, neg_counts = timeline.count_batch(batch_data, neg_arr, width)
            b_pos, b_neg = timeline.score_batch(rels, pos_counts, neg_counts, b_cfg)

            neg_valid = a_mask & c_mask & neg_mask[:, :width]
            valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), neg_valid), axis=1)
            cand_ids = np.concatenate((batch_data[:, 2:3], neg_arr[:, :width]), axis=1)
            a_all = np.concatenate((a_pos, a_neg), axis=1).astype(np.float32, copy=False)
            b_all = np.concatenate((b_pos, b_neg), axis=1).astype(np.float32, copy=False)
            c_all = np.concatenate((c_pos, c_neg), axis=1).astype(np.float32, copy=False)
            b_counts = np.concatenate((pos_counts.reshape(-1, 1), neg_counts), axis=1).astype(np.float32, copy=False)
            base_all = base_scores(a_all, b_all, c_all, valid)

            scores = {"a": a_all, "b": b_all, "c": c_all, "base": base_all}
            base_ranks = dense_rank(base_all, valid)
            base_loose, base_strict, base_avg = candidate_rank_pair(base_all, valid)
            cube = feature_builder.make(batch_data, cand_ids, valid, scores, b_counts, history)
            yield batch_data, valid, base_ranks, base_loose, base_strict, base_avg, cube
            row_offset = end

        update_events = (
            events
            if split == "train"
            else events_for_update(events, data, bool(getattr(args, "close_update_backward", False)))
        )
        timeline.update(update_events)
        history.update(update_events)

    if row_offset != a_store.num_rows or row_offset != c_store.num_rows:
        raise ValueError(
            f"row count mismatch for {split}: stream={row_offset}, "
            f"A={a_store.num_rows}, C={c_store.num_rows}"
        )


def build_training_matrix(a_run, c_run, b_cfg, data, feature_builder, args):
    return build_ranker_matrix(a_run, c_run, b_cfg, data, feature_builder, "train", args.train_topk, args)


def build_validation_matrix(a_run, c_run, b_cfg, data, feature_builder, args):
    raise RuntimeError("validation metrics are evaluated in streaming mode; do not build a full validation matrix")


def eval_stream_args(args):
    values = vars(args).copy()
    values["block_size"] = int(getattr(args, "eval_batch_size", getattr(args, "block_size", 256)))
    return SimpleNamespace(**values)


def split_query_count(data, split):
    return int(sum(len(events) for events, _, _ in split_snapshots(data, split)))


def build_ranker_matrix(a_run, c_run, b_cfg, data, feature_builder, split, topk, args):
    X_parts = []
    y_parts = []
    groups = []
    num_queries = 0
    num_rows = 0

    iterator = iter_feature_blocks(a_run, c_run, b_cfg, data, split, feature_builder, args)
    for _, valid, base_ranks, _, _, _, cube in tqdm(iterator, desc=f"build_{split}", leave=False):
        selected = valid.copy() if topk is None else selected_for_training(base_ranks, valid, topk)
        for row in range(selected.shape[0]):
            cols = np.flatnonzero(selected[row])
            if len(cols) <= 1:
                continue
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[np.flatnonzero(cols == 0)[0]] = 1.0
            X_parts.append(cube[row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            num_queries += 1
            num_rows += len(cols)

    if not X_parts:
        raise ValueError(f"no LGBM {split} rows built; check train_predict_ratio/topk")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {"queries": int(num_queries), "rows": int(num_rows)},
    )


def predict_lgbm(model, X):
    return model.predict(X).astype(np.float32, copy=False)


def evaluate_model(a_run, c_run, b_cfg, data, feature_builder, model, split, args):
    sums = {}
    stream_args = eval_stream_args(args)
    iterator = iter_feature_blocks(a_run, c_run, b_cfg, data, split, feature_builder, stream_args)
    for _, valid, _, _, _, _, cube in tqdm(iterator, desc=f"eval_{split}", leave=False):
        loose_ranks = []
        strict_ranks = []
        avg_ranks = []
        X_parts = []
        slices = []
        cursor = 0

        for row in range(valid.shape[0]):
            cols = np.flatnonzero(valid[row])
            start = cursor
            X_parts.append(cube[row, cols, :])
            cursor += len(cols)
            slices.append((start, cursor, 0))

        if X_parts:
            X = np.vstack(X_parts).astype(np.float32, copy=False)
            pred = predict_lgbm(model, X)
            for start, end, pos_idx in slices:
                scores = pred[start:end]
                pos_score = scores[pos_idx]
                other = np.delete(scores, pos_idx)
                loose = 1 + int(np.sum(other > pos_score))
                strict = 1 + int(np.sum(other >= pos_score))
                loose_ranks.append(loose)
                strict_ranks.append(strict)
                avg_ranks.append((loose + strict) * 0.5)

        add_rank_sums(
            sums,
            np.asarray(loose_ranks, dtype=np.int64),
            np.asarray(strict_ranks, dtype=np.int64),
            np.asarray(avg_ranks, dtype=np.float64),
        )
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


def evaluate_b_only(data, b_cfg, args):
    stream_args = eval_stream_args(args)
    close_update_backward = bool(getattr(args, "close_update_backward", False))
    timeline, _ = init_stream_state(data, "val", close_update_backward=close_update_backward)
    sums = {}
    desc = f"B_val_{b_cfg.mode}_close_update_backward{int(close_update_backward)}"
    for events, _, t_orig in tqdm(data["val_list"], desc=desc, leave=False):
        batches = collect_eval_batch(events, t_orig, data["negative_sampler"], "val", stream_args.block_size)
        for batch_data, neg_arr, neg_mask in batches:
            rels, pos_counts, neg_counts = timeline.count_batch(batch_data, neg_arr, neg_arr.shape[1])
            pos, neg = timeline.score_batch(rels, pos_counts, neg_counts, b_cfg)
            add_metric_sums(sums, compute_ranking_metric_sums(pos, neg, neg_mask))
        update_events = events_for_update(events, data, close_update_backward)
        timeline.update(update_events)
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


def sample_lgbm_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
    }


def default_lgbm_params(args):
    return {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "max_depth": int(getattr(args, "max_depth", -1)),
        "min_child_samples": int(args.min_child_samples),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(getattr(args, "reg_alpha", 0.0)),
        "min_split_gain": float(getattr(args, "min_split_gain", 0.0)),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
    }


def fit_lgbm(X, y, group, feature_builder, args, params=None, eval_data=None):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("combine_lgbm.py requires lightgbm to train the final combiner") from exc

    params = default_lgbm_params(args) if params is None else params
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
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
    fit_kwargs = {
        "group": group.tolist(),
        "feature_name": feature_builder.feature_names,
        "categorical_feature": feature_builder.categorical_indices,
    }
    if eval_data is not None:
        X_val, y_val, group_val = eval_data
        fit_kwargs.update(
            {
                "eval_set": [(X_val, y_val)],
                "eval_group": [group_val.tolist()],
                "eval_at": [10],
                "callbacks": [
                    lgb.early_stopping(
                        int(getattr(args, "lgbm_early_stopping_rounds", 50)),
                        verbose=False,
                    )
                ],
            }
        )
    model.fit(X, y, **fit_kwargs)
    return model


def tune_lgbm(a_run, c_run, b_cfg, data, feature_builder, X_train, y_train, group, args):
    n_trials = int(getattr(args, "lgbm_n_trials", 30))
    if n_trials <= 0:
        params = default_lgbm_params(args)
        model = fit_lgbm(
            X_train,
            y_train,
            group,
            feature_builder,
            args,
            params=params,
            eval_data=None,
        )
        val_metrics = evaluate_model(a_run, c_run, b_cfg, data, feature_builder, model, "val", args)
        score = metric_value(val_metrics, args.metric)
        best_iteration = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
        print(
            f"[combine] fixed LGBM val_{ranking_metric_key(args.metric)}={score:.5f} "
            f"best_iteration={best_iteration} params={params}",
            flush=True,
        )
        return model, {
            "n_trials": 0,
            "best_trial": None,
            "best_score": float(score),
            "best_params": params,
            "best_iteration": best_iteration,
            "val_metrics": val_metrics,
            "search": "fixed",
        }

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("Optuna is required for validation-based LGBM parameter selection") from exc

    best = {"score": -float("inf"), "model": None, "params": None, "val_metrics": None}
    direction = "maximize"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial):
        params = sample_lgbm_params(trial)
        model = fit_lgbm(
            X_train,
            y_train,
            group,
            feature_builder,
            args,
            params=params,
            eval_data=None,
        )
        val_metrics = evaluate_model(a_run, c_run, b_cfg, data, feature_builder, model, "val", args)
        score = metric_value(val_metrics, args.metric)
        trial.set_user_attr("val_metrics", val_metrics)
        trial.set_user_attr("best_iteration", int(getattr(model, "best_iteration_", 0) or params["n_estimators"]))
        if score > best["score"]:
            if best["model"] is not None:
                del best["model"]
                gc.collect()
            best.update(
                {
                    "score": float(score),
                    "model": model,
                    "params": dict(params),
                    "val_metrics": val_metrics,
                    "best_iteration": int(getattr(model, "best_iteration_", 0) or params["n_estimators"]),
                }
            )
        else:
            del model
            gc.collect()
        return score

    study.optimize(objective, n_trials=n_trials)
    if best["model"] is None:
        raise RuntimeError("LGBM tuning failed to produce a model")
    print(
        f"[combine] best LGBM trial={study.best_trial.number} "
        f"val_{ranking_metric_key(args.metric)}={best['score']:.5f} "
        f"best_iteration={best['best_iteration']} params={best['params']}",
        flush=True,
    )
    return best["model"], {
        "n_trials": n_trials,
        "best_trial": int(study.best_trial.number),
        "best_score": float(best["score"]),
        "best_params": best["params"],
        "best_iteration": int(best["best_iteration"]),
        "val_metrics": best["val_metrics"],
    }


def validate_args(args):
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if not 0.0 < float(args.train_predict_ratio) <= 1.0:
        raise ValueError("--train_predict_ratio must be in (0, 1] for LGBM training")
    if int(args.train_topk) <= 0:
        raise ValueError("--train_topk must be > 0")
    if int(getattr(args, "eval_batch_size", getattr(args, "block_size", 256))) <= 0:
        raise ValueError("--eval_batch_size must be > 0")
    if int(args.top_a) <= 0 or int(args.top_b) <= 0 or int(args.top_c) <= 0:
        raise ValueError("--top_a/top_b/top_c must be > 0")
    ranking_metric_key(args.metric, strict=True)


def run_search(args):
    validate_args(args)
    set_random_seed(args.seed)
    start_time = time.time()
    out_dir = make_out_dir(args)
    ensure_dir(out_dir)
    print(f"[combine] output -> {out_dir}", flush=True)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    if not data["train_predict_count"]:
        raise ValueError("train_predict_ratio selected no training timestamps")

    a_runs = [load_single_score_run(args.a_dir, "A", args)]
    c_runs = [load_single_score_run(args.c_dir, "C", args)]
    b_cfg = single_b_config(args)
    b_metrics = evaluate_b_only(data, b_cfg, args)
    b_records = [{"config": b_cfg, "metrics": b_metrics, "score": metric_value(b_metrics, args.component_metric)}]
    print(
        f"[combine] single A={a_runs[0]['dir']} C={c_runs[0]['dir']} "
        f"B={asdict(b_cfg)} {ranking_metric_key(args.component_metric)}={b_records[0]['score']:.5f}",
        flush=True,
    )
    combos = [(a_run, c_run, b_rec) for a_run in a_runs for c_run in c_runs for b_rec in b_records]

    print(
        f"[combine] A={len(a_runs)} B={len(b_records)} C={len(c_runs)} combos={len(combos)} "
        f"selection_metric={ranking_metric_key(args.metric)}",
        flush=True,
    )

    feature_builder = FeatureBuilder(data["num_rels"])
    candidates = []
    best = None

    for idx, (a_run, c_run, b_rec) in enumerate(combos, start=1):
        b_cfg = b_rec["config"]
        combo_key = make_combo_key(args, a_run, c_run, b_cfg)
        print(
            f"\n[combine] combo {idx}/{len(combos)} "
            f"A={osp.basename(a_run['dir'])} C={osp.basename(c_run['dir'])} B={asdict(b_cfg)}",
            flush=True,
        )
        cached = load_combo_cache(out_dir, combo_key)
        if cached is not None:
            record, model_path = cached
            cached_model = None
            if best is None or record["val_score"] > best["record"]["val_score"]:
                try:
                    cached_model = load_lgbm_model(model_path)
                except Exception as exc:
                    print(f"[combine] skip combo cache {combo_key}: cannot load model: {exc}", flush=True)
                    cached = None
            if cached is not None:
                candidates.append(record)
                print(f"[combine] cache hit combo={combo_key}", flush=True)
                print(f"[combine] val {format_metrics(record['val_metrics'])}", flush=True)
                if cached_model is not None:
                    old_best = best
                    best = {
                        "record": record,
                        "model": cached_model,
                        "a_run": a_run,
                        "c_run": c_run,
                        "b_cfg": b_cfg,
                    }
                    if old_best is not None:
                        del old_best["model"]
                        gc.collect()
                continue

        X_train, y_train, group, train_info = build_training_matrix(
            a_run, c_run, b_cfg, data, feature_builder, args
        )
        print(
            f"[combine] train rows={train_info['rows']} queries={train_info['queries']} "
            f"features={X_train.shape[1]}",
            flush=True,
        )
        val_info = {
            "mode": "full_metric_stream",
            "queries": split_query_count(data, "val"),
            "eval_batch_size": int(getattr(args, "eval_batch_size", getattr(args, "block_size", 256))),
        }
        print(
            f"[combine] val metric stream queries={val_info['queries']} "
            f"eval_batch_size={val_info['eval_batch_size']}",
            flush=True,
        )
        model, lgbm_tuning = tune_lgbm(
            a_run,
            c_run,
            b_cfg,
            data,
            feature_builder,
            X_train,
            y_train,
            group,
            args,
        )
        del X_train, y_train, group
        gc.collect()

        val_metrics = lgbm_tuning["val_metrics"]
        val_score = metric_value(val_metrics, args.metric)
        record = {
            "val_score": float(val_score),
            "val_metrics": val_metrics,
            "train_info": train_info,
            "val_eval_info": val_info,
            "val_tune_info": val_info,
            "a_dir": a_run["dir"],
            "c_dir": c_run["dir"],
            "a_config": short_config(a_run["config"]),
            "c_config": short_config(c_run["config"]),
            "b_config": asdict(b_cfg),
            "b_val_metrics": b_rec["metrics"],
            "lgbm_tuning": lgbm_tuning,
            "combo_key": combo_key,
        }
        candidates.append(record)
        print(f"[combine] val {format_metrics(val_metrics)}", flush=True)
        save_combo_cache(out_dir, combo_key, record, model)

        if best is None or val_score > best["record"]["val_score"]:
            old_best = best
            best = {"record": record, "model": model, "a_run": a_run, "c_run": c_run, "b_cfg": b_cfg}
            if old_best is not None:
                del old_best["model"]
                gc.collect()
        else:
            del model
            gc.collect()

    candidates.sort(key=lambda item: item["val_score"], reverse=True)
    best_record = best["record"]
    print(
        f"\n[combine] best val {ranking_metric_key(args.metric)}={best_record['val_score']:.5f}",
        flush=True,
    )
    print(f"[combine] best A config: {best_record['a_config']}", flush=True)
    print(f"[combine] best B config: {best_record['b_config']}", flush=True)
    print(f"[combine] best C config: {best_record['c_config']}", flush=True)

    test_metrics = None
    if getattr(args, "eval_test", True):
        test_metrics = evaluate_model(
            best["a_run"], best["c_run"], best["b_cfg"], data, feature_builder, best["model"], "test", args
        )
        best_record["test_metrics"] = test_metrics
        print(f"[combine] final test {format_metrics(test_metrics)}", flush=True)
        save_combo_cache(out_dir, best_record["combo_key"], best_record, best["model"])

    model_path = osp.join(out_dir, "best_lgbm.txt")
    save_lgbm_model(best["model"], model_path)

    summary = {
        "format": "abc_lgbm_v1",
        "dataset": args.dataset,
        "args": vars(args),
        "selection_metric": ranking_metric_key(args.metric),
        "component_preselect_metric": ranking_metric_key(args.component_metric),
        "feature_names": feature_builder.feature_names,
        "categorical_features": feature_builder.categorical_names,
        "best": {
            **best_record,
            "model_path": model_path,
        },
        "top_by_validation": candidates[: int(args.print_top)],
        "runtime_sec": float(time.time() - start_time),
    }
    save_config(out_dir, vars(args))
    save_metrics(out_dir, summary)
    with open(osp.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[combine] saved -> {out_dir}", flush=True)
    return summary
