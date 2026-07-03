import math
import time

import numpy as np
from tqdm import tqdm

from utils import (
    TGB_DATASETS,
    ScoreWriter,
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    describe_loaded_data,
    finalize_metric_sums,
    inverse_aug,
    is_run_complete,
    load_datasets,
    load_metrics,
    make_dir_name,
    save_config,
    save_metrics,
    set_random_seed,
)


class APredictor:
    def __init__(self, num_nodes, num_rels, a_mode="energy", decay_a=1.0, ppr_beta=0.8):
        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.a_mode = a_mode
        self.decay_a = float(decay_a)
        self.ppr_beta = float(ppr_beta)
        self.rel_shift = 10 ** int(math.ceil(math.log10(self.num_rels + 1)))
        self.V_sr = {}
        self.V_sr_sum = {}
        self.time_shift_a = 0.0

    def _sr_key(self, s, r):
        return int(s) * self.rel_shift + int(r)

    def update_state(self, events, ts):
        if self.a_mode == "energy":
            self._update_energy(events, ts)
        elif self.a_mode == "rank":
            self._update_rank(events)
        else:
            raise ValueError(f"unknown a_mode: {self.a_mode}")

    def _update_energy(self, events, ts):
        rel_exp = self.decay_a * (float(ts) - self.time_shift_a)
        if rel_exp > 700.0:
            scale = 2.0 ** (-rel_exp)
            for key in self.V_sr_sum:
                self.V_sr_sum[key] *= scale
            for inner in self.V_sr.values():
                for obj in inner:
                    inner[obj] *= scale
            self.time_shift_a = float(ts)
            weight = 1.0
        else:
            weight = 2.0 ** rel_exp

        for s, r, o in events[:, :3].astype(np.int64, copy=False):
            sr = self._sr_key(s, r)
            bucket = self.V_sr.setdefault(sr, {})
            bucket[int(o)] = bucket.get(int(o), 0.0) + weight
            self.V_sr_sum[sr] = self.V_sr_sum.get(sr, 0.0) + weight

    def _update_rank(self, events):
        affected = {self._sr_key(s, r) for s, r in events[:, :2].astype(np.int64, copy=False)}
        for sr in affected:
            if sr not in self.V_sr:
                continue
            for obj in list(self.V_sr[sr].keys()):
                self.V_sr[sr][obj] *= self.ppr_beta
            self.V_sr_sum[sr] *= self.ppr_beta

        for s, r, o in events[:, :3].astype(np.int64, copy=False):
            sr = self._sr_key(s, r)
            bucket = self.V_sr.setdefault(sr, {})
            bucket[int(o)] = bucket.get(int(o), 0.0) + 1.0
            self.V_sr_sum[sr] = self.V_sr_sum.get(sr, 0.0) + 1.0

    def predict_batch(self, batch_data, neg_samples):
        batch_size, width = neg_samples.shape
        pos = np.zeros((batch_size, 1), dtype=np.float32)
        neg = np.zeros((batch_size, width), dtype=np.float32)

        for i, (s, r, o) in enumerate(batch_data[:, :3].astype(np.int64, copy=False)):
            sr = self._sr_key(s, r)
            bucket = self.V_sr.get(sr)
            total = self.V_sr_sum.get(sr, 0.0)
            if not bucket or total <= 0.0:
                continue

            inv_total = 1.0 / total
            pos[i, 0] = bucket.get(int(o), 0.0) * inv_total
            for j in np.flatnonzero(neg_samples[i] != -1):
                neg[i, j] = bucket.get(int(neg_samples[i, j]), 0.0) * inv_total

        return pos, neg


def build_predictor(args, data):
    return APredictor(
        num_nodes=data["num_nodes"],
        num_rels=data["num_rels"],
        a_mode=args.a_mode,
        decay_a=args.decay_a,
        ppr_beta=args.ppr_beta,
    )


def should_inverse_aug(data, args):
    return data["is_tgb"] and not bool(getattr(args, "close_update_backward", False))


def eval_snapshot_list(predictor, snapshots, data, args, out_dir, mode, update_after):
    writer = ScoreWriter(out_dir, mode)
    metric_sums = {}
    start_time = time.time()

    for events, t_norm, t_orig in tqdm(snapshots, desc=mode):
        batches = collect_eval_batch(events, t_orig, data["negative_sampler"], mode, args.batch_size)
        for batch, neg_arr, neg_mask in batches:
            pos, neg = predictor.predict_batch(batch, neg_arr)
            batch_sums = compute_ranking_metric_sums(pos, neg, neg_mask)
            add_metric_sums(metric_sums, batch_sums)
            writer.write_batch(pos, neg, neg_mask)

        update_events = update_after(events) if update_after is not None else events
        predictor.update_state(update_events, t_norm)

    writer.close()
    metrics = finalize_metric_sums(metric_sums)
    print(f"[A] {mode} time: {time.time() - start_time:.1f}s", flush=True)
    print(
        f"[A] {mode} "
        f"mrr_loose={metrics['mrr_loose']:.5f} "
        f"mrr_strict={metrics['mrr_strict']:.5f} "
        f"mrr_avg={metrics['mrr_avg']:.5f}",
        flush=True,
    )
    return metrics


def run_stream(predictor, data, args, out_dir):
    train_list = data["train_list"]
    train_predict_start_idx = data["train_predict_start_idx"]
    train_warmup = train_list[:train_predict_start_idx]
    train_predict = train_list[train_predict_start_idx:]

    print(f"[A] warmup train snapshots: {len(train_warmup)}", flush=True)
    for events, t_norm, _ in tqdm(train_warmup, desc="train"):
        predictor.update_state(events, t_norm)

    metrics = {}
    if train_predict:
        print(f"[A] predict-then-train snapshots: {len(train_predict)}", flush=True)
        train_metrics = eval_snapshot_list(
            predictor,
            train_predict,
            data,
            args,
            out_dir,
            "train",
            update_after=None,
        )
        metrics.update({f"train_{key}": value for key, value in train_metrics.items()})

    def eval_update(events):
        if should_inverse_aug(data, args):
            return inverse_aug(events, data["num_rels_raw"], data["num_rels"])
        return events

    val_metrics = eval_snapshot_list(predictor, data["val_list"], data, args, out_dir, "val", eval_update)
    metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
    if getattr(args, "eval_test", True):
        test_metrics = eval_snapshot_list(predictor, data["test_list"], data, args, out_dir, "test", eval_update)
        metrics.update({f"test_{key}": value for key, value in test_metrics.items()})
    return metrics


def get_out_dir(args):
    params = dict(
        a_mode=args.a_mode,
        decay_a=args.decay_a,
        ppr_beta=args.ppr_beta,
        ns_q=args.ns_q,
        ns_seed=args.ns_seed,
        train_predict_ratio=args.train_predict_ratio,
    )
    if args.dataset in TGB_DATASETS:
        params["close_update_backward"] = bool(getattr(args, "close_update_backward", False))
    return make_dir_name("results_a_single", args.dataset, args.seed, **params)


def validate_args(args):
    if not 0.0 <= args.train_predict_ratio <= 1.0:
        raise ValueError("--train_predict_ratio must be in [0, 1]")
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")


def main(args):
    validate_args(args)
    set_random_seed(args.seed)

    data = load_datasets(args.dataset, q=args.ns_q, load_train_ratio=args.train_predict_ratio, ns_seed=args.ns_seed)
    describe_loaded_data(data, prefix="[A]")

    out_dir = get_out_dir(args)
    expected_modes = ("train", "val") if data["train_predict_count"] else ("val",)
    if getattr(args, "eval_test", True):
        expected_modes = expected_modes + ("test",)
    if is_run_complete(out_dir, expected_modes):
        metrics = load_metrics(out_dir)
        has_required = "val_mrr_avg" in metrics
        if getattr(args, "eval_test", True):
            has_required = has_required and "test_mrr_loose" in metrics and "test_mrr_strict" in metrics
        if has_required:
            metrics["val_mrr"] = metrics["val_mrr_strict"]
            if "test_mrr_strict" in metrics:
                metrics["test_mrr"] = metrics["test_mrr_strict"]
            print(f"[A] already complete: {out_dir}", flush=True)
            msg = (
                f"[A] val_mrr_loose={metrics['val_mrr_loose']:.5f} "
                f"val_mrr_strict={metrics['val_mrr_strict']:.5f} "
                f"val_mrr_avg={metrics['val_mrr_avg']:.5f}"
            )
            if "test_mrr_strict" in metrics:
                msg += (
                    f" test_mrr_loose={metrics['test_mrr_loose']:.5f} "
                    f"test_mrr_strict={metrics['test_mrr_strict']:.5f} "
                    f"test_mrr_avg={metrics['test_mrr_avg']:.5f}"
                )
            print(msg, flush=True)
            return metrics
        print(f"[A] stale metrics found, recomputing: {out_dir}", flush=True)

    print(f"[A] running -> {out_dir}", flush=True)
    predictor = build_predictor(args, data)
    metrics = run_stream(predictor, data, args, out_dir)
    metrics["val_mrr"] = metrics["val_mrr_strict"]
    msg = (
        f"[A] val_mrr_loose={metrics['val_mrr_loose']:.5f} "
        f"val_mrr_strict={metrics['val_mrr_strict']:.5f} "
        f"val_mrr_avg={metrics['val_mrr_avg']:.5f}"
    )
    if "test_mrr_strict" in metrics:
        metrics["test_mrr"] = metrics["test_mrr_strict"]
        msg += (
            f" test_mrr_loose={metrics['test_mrr_loose']:.5f} "
            f"test_mrr_strict={metrics['test_mrr_strict']:.5f} "
            f"test_mrr_avg={metrics['test_mrr_avg']:.5f}"
        )
    print(msg, flush=True)

    save_config(out_dir, vars(args))
    save_metrics(out_dir, metrics)
    return metrics


def tune_a(args, grid=None, top_k=3, metric="val_mrr_strict"):
    """Run a small transparent grid and rank A configs by validation only."""
    from copy import deepcopy

    if grid is None:
        grid = [
            {"a_mode": "rank", "decay_a": 1.0, "ppr_beta": beta}
            for beta in (0.8, 0.9, 0.95, 0.98)
        ] + [
            {"a_mode": "energy", "decay_a": decay_a, "ppr_beta": 0.9}
            for decay_a in (0.1, 0.5, 1.0, 2.0)
        ]

    records = []
    for idx, params in enumerate(grid, start=1):
        run_args = deepcopy(args)
        for key, value in params.items():
            setattr(run_args, key, value)
        setattr(run_args, "eval_test", False)
        metrics = main(run_args)
        out_dir = get_out_dir(run_args)
        score = float(metrics[metric])
        records.append(
            {
                "rank_source": "validation",
                "score": score,
                "metric": metric,
                "params": dict(params),
                "out_dir": out_dir,
                "args": vars(run_args).copy(),
            }
        )
        print(f"[A-tune] {idx}/{len(grid)} val {metric}={score:.5f} params={params}", flush=True)

    records.sort(key=lambda item: item["score"], reverse=True)
    return records[: int(top_k)]
