import json
import os
import os.path as osp
from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np

from single_pipeline import a_single, c_single, hybrid_single, structure_combine_single, time_single
from single_pipeline.structure_combine_single import BConfig
from utils import (
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    finalize_metric_sums,
    load_datasets,
    load_metrics,
    ranking_metric_key,
)


def clone_args(args, **overrides):
    values = vars(args).copy() if hasattr(args, "__dict__") else dict(args)
    values.update(overrides)
    return SimpleNamespace(**values)


def save_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def metric_token(metric):
    key = ranking_metric_key(metric, strict=True)
    if key == "mrr_strict":
        return "mrr"
    if key.startswith("hit@") and key.endswith("_strict"):
        return "h" + key[len("hit@") : -len("_strict")]
    return key.replace("@", "").replace("_strict", "")


def protocol_dir(args):
    return (
        f"p-nq{args.ns_q}-ns{args.ns_seed}"
        f"_tr{args.train_predict_ratio:g}"
        f"_cub{int(bool(getattr(args, 'close_update_backward', False)))}"
    )


def search_dir(args, metric):
    return osp.join(
        protocol_dir(args),
        f"m-{metric_token(metric)}",
        (
            f"k-a{getattr(args, 'top_k_a', 0)}"
            f"-b{getattr(args, 'top_k_b', 0)}"
            f"-c{getattr(args, 'top_k_c', 0)}"
            f"-tt{getattr(args, 'top_k_time', 0)}"
            f"-ts{getattr(args, 'top_k_struct', 0)}"
            f"_tk{getattr(args, 'train_topk', 100)}"
            f"_l{getattr(args, 'lgbm_n_trials', 30)}"
            f"_e{getattr(args, 'lgbm_early_stopping_rounds', 50)}"
        ),
    )


def strict_val_score(metrics, metric="mrr"):
    key = f"val_{ranking_metric_key(metric, strict=True)}"
    if key in metrics:
        return float(metrics[key])
    if metric in ("mrr", "MRR"):
        return float(metrics.get("val_mrr", metrics.get("val_mrr_strict", 0.0)))
    if metric in ("hr10", "hit10", "hit@10"):
        return float(metrics.get("val_hit10", metrics.get("val_hit@10_strict", 0.0)))
    return float(metrics[key])


def default_struct_args(base_args, a_dir, c_dir, b_params, eval_test=False):
    return clone_args(
        base_args,
        metric=getattr(base_args, "metric", "hr10"),
        component_metric=getattr(base_args, "component_metric", "mrr"),
        a_prefix="results_a",
        c_prefix="results_c",
        out_prefix="results_lgbm_single",
        a_dir=a_dir,
        c_dir=c_dir,
        top_a=1,
        top_b=1,
        top_c=1,
        close_update_backward=bool(getattr(base_args, "close_update_backward", False)),
        block_size=getattr(base_args, "block_size", 256),
        train_topk=getattr(base_args, "train_topk", 100),
        eval_batch_size=getattr(base_args, "eval_batch_size", getattr(base_args, "block_size", 256)),
        b_modes="binary,continuous",
        b_mode=b_params["mode"],
        b_binary_unseen=float(b_params.get("binary_unseen", 0.0)),
        b_continuous_alpha=float(b_params.get("continuous_alpha", 0.0001)),
        binary_unseen_grid="0",
        continuous_alpha_grid="0.0001",
        n_estimators=getattr(base_args, "n_estimators", 1000),
        learning_rate=getattr(base_args, "learning_rate", 0.03),
        num_leaves=getattr(base_args, "num_leaves", 63),
        min_child_samples=getattr(base_args, "min_child_samples", 50),
        reg_lambda=getattr(base_args, "reg_lambda", 1.0),
        reg_alpha=getattr(base_args, "reg_alpha", 0.0),
        max_depth=getattr(base_args, "max_depth", -1),
        min_split_gain=getattr(base_args, "min_split_gain", 0.0),
        subsample=getattr(base_args, "subsample", 0.9),
        colsample_bytree=getattr(base_args, "colsample_bytree", 0.9),
        lgbm_n_trials=getattr(base_args, "lgbm_n_trials", 30),
        lgbm_early_stopping_rounds=getattr(base_args, "lgbm_early_stopping_rounds", 50),
        num_threads=getattr(base_args, "num_threads", 8),
        print_top=getattr(base_args, "print_top", 20),
        eval_test=eval_test,
    )


def default_hybrid_args(base_args, struct_dir, struct_combo_key, time_dir, eval_test=False):
    args = clone_args(
        base_args,
        metric=getattr(base_args, "metric", "hr10"),
        struct_metric=getattr(base_args, "struct_metric", ""),
        time_metric=getattr(base_args, "time_metric", ""),
        struct_prefix="results_lgbm_single",
        time_prefix="results_time_tkg_single",
        out_prefix="results_hybrid_lgbm_single",
        struct_dir=struct_dir,
        struct_combo_key=struct_combo_key,
        time_dir=time_dir,
        block_size=getattr(base_args, "block_size", 128),
        top_k_struct=1,
        top_k_time=1,
        train_topk=getattr(base_args, "train_topk", 100),
        eval_batch_size=getattr(base_args, "eval_batch_size", getattr(base_args, "block_size", 128)),
        n_estimators=getattr(base_args, "n_estimators", 1000),
        learning_rate=getattr(base_args, "learning_rate", 0.03),
        num_leaves=getattr(base_args, "num_leaves", 63),
        min_child_samples=getattr(base_args, "min_child_samples", 50),
        reg_lambda=getattr(base_args, "reg_lambda", 1.0),
        reg_alpha=getattr(base_args, "reg_alpha", 0.0),
        max_depth=getattr(base_args, "max_depth", -1),
        min_split_gain=getattr(base_args, "min_split_gain", 0.0),
        subsample=getattr(base_args, "subsample", 0.9),
        colsample_bytree=getattr(base_args, "colsample_bytree", 0.9),
        lgbm_n_trials=getattr(base_args, "lgbm_n_trials", 30),
        lgbm_early_stopping_rounds=getattr(base_args, "lgbm_early_stopping_rounds", 50),
        num_threads=getattr(base_args, "num_threads", 8),
        force=getattr(base_args, "force_hybrid", False),
        eval_test=eval_test,
    )
    if hasattr(args, "close_update_backward"):
        delattr(args, "close_update_backward")
    return args


def tune_b(base_args, grid=None, top_k=3, metric="mrr", out_dir=None):
    """Precompute raw B counts once, then score a small B grid on validation."""
    if grid is None:
        grid = [
            {"mode": "binary", "binary_unseen": v, "continuous_alpha": 0.0001}
            for v in (0.0, 0.001, 0.01)
        ] + [
            {"mode": "continuous", "binary_unseen": 0.0, "continuous_alpha": a}
            for a in (0.0001, 0.001, 0.01, 0.1)
        ]
    if out_dir is None:
        out_dir = osp.join("results_b_single", base_args.dataset, f"seed{base_args.seed}")
    raw_path = osp.join(
        out_dir,
        (
            f"val_counts_nq{base_args.ns_q}_ns{base_args.ns_seed}"
            f"_tr{base_args.train_predict_ratio:g}"
            f"_cub{int(bool(getattr(base_args, 'close_update_backward', False)))}.npz"
        ),
    )
    if not osp.isfile(raw_path):
        precompute_b_val_counts(base_args, raw_path)

    raw = np.load(raw_path)
    records = []
    for params in grid:
        metrics = evaluate_b_grid_params(raw, params, num_nodes=int(raw["num_nodes"]))
        score = strict_val_score({f"val_{k}": v for k, v in metrics.items()}, metric)
        stored_params = dict(params)
        records.append(
            {
                "rank_source": "validation",
                "score": score,
                "metric": f"val_{ranking_metric_key(metric, strict=True)}",
                "params": stored_params,
                "raw_count_path": raw_path,
            }
        )
        print(f"[B-tune] val {metric}={score:.5f} params={params}", flush=True)
    records.sort(key=lambda item: item["score"], reverse=True)
    save_json(osp.join(out_dir, "tune_b_summary.json"), {"top_by_validation": records})
    return records[: int(top_k)]


def precompute_b_val_counts(args, raw_path):
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    close_update_backward = bool(getattr(args, "close_update_backward", False))
    b_cfg = BConfig("continuous", 0.0, 0.0001)
    timeline, _ = structure_combine_single.init_stream_state(
        data,
        "val",
        close_update_backward=close_update_backward,
    )
    rels_all, pos_all, neg_all, denom_all, lens_all = [], [], [], [], []
    for events, _, t_orig in data["val_list"]:
        batches = collect_eval_batch(events, t_orig, data["negative_sampler"], "val", getattr(args, "block_size", 256))
        for batch_data, neg_arr, neg_mask in batches:
            rels, pos_counts, neg_counts = timeline.count_batch(batch_data, neg_arr, neg_arr.shape[1])
            rels_all.append(rels.astype(np.int64, copy=False))
            pos_all.append(pos_counts.astype(np.float32, copy=False))
            neg_all.append(neg_counts.astype(np.float32, copy=False))
            denom_all.append(timeline.counts_r[rels].astype(np.float32, copy=False))
            lens_all.append(neg_mask.sum(axis=1).astype(np.int32, copy=False))
        timeline.update(structure_combine_single.events_for_update(events, data, close_update_backward))
    os.makedirs(osp.dirname(raw_path), exist_ok=True)
    np.savez_compressed(
        raw_path,
        rels=np.concatenate(rels_all),
        pos_counts=np.concatenate(pos_all),
        neg_counts=np.vstack(neg_all),
        rel_totals=np.concatenate(denom_all),
        valid_lens=np.concatenate(lens_all),
        num_nodes=np.asarray(data["num_nodes"], dtype=np.int64),
    )
    print(f"[B-tune] saved raw validation counts -> {raw_path}", flush=True)


def evaluate_b_grid_params(raw, params, num_nodes):
    pos_counts = raw["pos_counts"].astype(np.float32)
    neg_counts = raw["neg_counts"].astype(np.float32)
    valid_lens = raw["valid_lens"].astype(np.int32)
    width = neg_counts.shape[1]
    valid = np.arange(width)[None, :] < valid_lens[:, None]
    if params["mode"] == "binary":
        unseen = float(params.get("binary_unseen", 0.0))
        pos = np.where(pos_counts > 0, 1.0, unseen).astype(np.float32).reshape(-1, 1)
        neg = np.where(neg_counts > 0, 1.0, unseen).astype(np.float32)
    else:
        alpha = float(params.get("continuous_alpha", 0.0001))
        denom = np.maximum(raw["rel_totals"].astype(np.float32) + alpha * int(num_nodes), 1e-12)
        pos = ((pos_counts + alpha) / denom).astype(np.float32).reshape(-1, 1)
        neg = ((neg_counts + alpha) / denom.reshape(-1, 1)).astype(np.float32)
    sums = {}
    add_metric_sums(sums, compute_ranking_metric_sums(pos, neg, valid))
    return finalize_metric_sums(sums)


def candidate_key(*items):
    return tuple(item.get("out_dir") or json.dumps(item.get("params", {}), sort_keys=True) for item in items)


def ensure_a_test(candidate):
    if "args" in candidate:
        args = clone_args(candidate["args"], eval_test=True)
        return a_single.main(args)
    return None


def ensure_c_test(candidate):
    if "args" in candidate:
        args = clone_args(candidate["args"], eval_test=True)
        return c_single.main(args)
    return None


def ensure_time_test(candidate):
    if "args" in candidate:
        args = clone_args(candidate["args"], eval_test=True)
        return time_single.main(args)
    return None


def run_struct(base_args, a_candidates, b_candidates, c_candidates, top_k_a=3, top_k_b=3, top_k_c=10, metric="hr10"):
    """Validate one best triple plus one-component replacements; test every validation combo."""
    a_top = a_candidates[: int(top_k_a)]
    b_top = b_candidates[: int(top_k_b)]
    c_top = c_candidates[: int(top_k_c)]
    best_a, best_b, best_c = a_top[0], b_top[0], c_top[0]
    combos = [(best_a, best_b, best_c)]
    combos += [(a, best_b, best_c) for a in a_top[1:]]
    combos += [(best_a, b, best_c) for b in b_top[1:]]
    combos += [(best_a, best_b, c) for c in c_top[1:]]

    seen = set()
    records = []
    for a_rec, b_rec, c_rec in combos:
        key = candidate_key(a_rec, b_rec, c_rec)
        if key in seen:
            continue
        seen.add(key)
        args = default_struct_args(
            clone_args(base_args, metric=metric),
            a_rec["out_dir"],
            c_rec["out_dir"],
            b_rec["params"],
            eval_test=True,
        )
        ensure_a_test(a_rec)
        ensure_c_test(c_rec)
        summary = structure_combine_single.run_search(args)
        val_score = float(summary["best"]["val_score"])
        test_summary = summary.get("best", {}).get("test_metrics", {})
        records.append(
            {
                "rank_source": "validation",
                "score": val_score,
                "metric": summary["selection_metric"],
                "out_dir": structure_combine_single.make_out_dir(args),
                "combo_key": summary["best"]["combo_key"],
                "a": a_rec,
                "b": b_rec,
                "c": c_rec,
                "args": vars(args).copy(),
                "test_summary": test_summary,
            }
        )
        test_text = ""
        if test_summary:
            test_key = ranking_metric_key(metric, strict=True)
            test_text = f" test_{test_key}={float(test_summary.get(test_key, 0.0)):.5f}"
        print(f"[struct-run] val {summary['selection_metric']}={val_score:.5f}{test_text}", flush=True)

    records.sort(key=lambda item: item["score"], reverse=True)
    winner = records[0]
    summary = {"top_by_validation": records, "best": winner}
    save_json(
        osp.join(
            "rs_tune",
            base_args.dataset,
            f"seed{base_args.seed}",
            search_dir(base_args, metric),
            "run_struct_summary.json",
        ),
        summary,
    )
    return summary


def tune_time(base_args, top_k=4, metric="mrr"):
    fixed = {
        "num_epochs": 50,
        "patience": 5,
        "train_num_neg": 4,
        "train_sampler": "grouped_exact",
        "event_encoder": "mixer",
        "event_dim": 96,
        "hidden_dim": 192,
        "multi_windows": "7,30,90",
        "use_neighbor_id": True,
        "use_abs_time": True,
        "abs_time_periods": "1,7,30,365",
        "abs_time_use_raw": False,
        "use_query_gate": True,
        "query_gate_type": "channel",
        "use_rank_pos": True,
        "train_loss": "margin",
    }
    records = []
    grid = product((200,1000), (0.1, 0.2))
    for topk, dropout in grid:
        args = clone_args(
            base_args,
            **fixed,
            topk=topk,
            num_layers=1,
            dropout=dropout,
            eval_test=False,
        )
        time_single.main(args)
        out_dir = time_single.get_out_dir(args)
        metrics = load_metrics(out_dir)
        score = strict_val_score(metrics, metric)
        records.append(
            {
                "rank_source": "validation",
                "score": score,
                "metric": f"val_{ranking_metric_key(metric, strict=True)}",
                "params": {"topk": topk, "num_layers": 1, "dropout": dropout, **fixed},
                "out_dir": out_dir,
                "args": vars(args).copy(),
            }
        )
        print(f"[time-tune] val {metric}={score:.5f} topk={topk} layers=fixed_1 dropout={dropout}", flush=True)
    records.sort(key=lambda item: item["score"], reverse=True)
    return records[: int(top_k)]


def run_hybrid(base_args, time_candidates, struct_candidates, top_k_time=8, top_k_struct=8, metric="hr10"):
    """Validate best-best plus one-side replacements; test only the best validation pair."""
    if isinstance(time_candidates, dict):
        time_candidates = time_candidates["top_by_validation"]
    if isinstance(struct_candidates, dict):
        struct_candidates = struct_candidates["top_by_validation"]
    time_top = time_candidates[: int(top_k_time)]
    struct_top = struct_candidates[: int(top_k_struct)]
    best_time, best_struct = time_top[0], struct_top[0]
    pairs = [(best_time, best_struct)]
    pairs += [(best_time, s) for s in struct_top[1:]]
    pairs += [(t, best_struct) for t in time_top[1:]]

    records = []
    seen = set()
    for time_rec, struct_rec in pairs:
        key = candidate_key(time_rec, struct_rec)
        if key in seen:
            continue
        seen.add(key)
        args = default_hybrid_args(
            clone_args(base_args, metric=metric),
            struct_rec["out_dir"],
            struct_rec["combo_key"],
            time_rec["out_dir"],
            eval_test=False,
        )
        summary = hybrid_single.run(args)
        val_score = float(summary["val_score"])
        records.append(
            {
                "rank_source": "validation",
                "score": val_score,
                "metric": summary["selection_metric"],
                "out_dir": osp.dirname(summary["model_path"]),
                "time": time_rec,
                "struct": struct_rec,
                "args": vars(args).copy(),
            }
        )
        print(f"[hybrid-run] val {summary['selection_metric']}={val_score:.5f}", flush=True)

    records.sort(key=lambda item: item["score"], reverse=True)
    winner = records[0]
    ensure_time_test(winner["time"])
    ensure_a_test(winner["struct"]["a"])
    ensure_c_test(winner["struct"]["c"])
    final_args = clone_args(winner["args"], eval_test=True, force=True)
    final_summary = hybrid_single.run(final_args)
    winner["test_summary"] = final_summary.get("test_metrics", {})
    save_json(
        osp.join(
            "rh_tune",
            base_args.dataset,
            f"seed{base_args.seed}",
            search_dir(base_args, metric),
            "run_hybrid_summary.json",
        ),
        {"top_by_validation": records, "best": winner},
    )
    return winner
