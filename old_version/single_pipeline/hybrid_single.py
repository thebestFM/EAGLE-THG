import gc
import glob
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from .structure_combine_single import (
    BConfig,
    FeatureBuilder,
    ScoreStore,
    add_rank_sums,
    candidate_rank_pair,
    combo_cache_dir,
    dense_rank,
    format_metrics,
    iter_feature_blocks,
    load_lgbm_model,
    metric_value,
    predict_lgbm,
    safe_token,
    stable_hash,
)
from utils import (
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


def score_from_metrics(metrics, metric, split="val"):
    key = f"{split}_{ranking_metric_key(metric, strict=True)}"
    if key in metrics:
        return float(metrics[key])
    metric_text = str(metric).lower().replace("@", "")
    if metric_text == "mrr" and f"{split}_mrr" in metrics:
        return float(metrics[f"{split}_mrr"])
    if metric_text in ("hr10", "hit10") and f"{split}_hit10" in metrics:
        return float(metrics[f"{split}_hit10"])
    return 0.0


def effective_struct_metric(args):
    return args.struct_metric or args.metric


def effective_time_metric(args):
    return args.time_metric or args.metric


def resolve_model_path(out_dir, path, fallback_name="best_lgbm.txt"):
    if path and osp.isfile(path):
        return path
    if path:
        joined = osp.join(out_dir, path)
        if osp.isfile(joined):
            return joined
    fallback = osp.join(out_dir, fallback_name)
    if osp.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"cannot find LightGBM model for {out_dir}")


def make_b_config(raw):
    return BConfig(
        mode=raw["mode"],
        binary_unseen=float(raw.get("binary_unseen", 0.0)),
        continuous_alpha=float(raw.get("continuous_alpha", 1.0)),
    )


def make_struct_candidate(out_dir, cfg, summary, record, model_path, score):
    return {
        "dir": out_dir,
        "config": cfg,
        "summary": summary,
        "record": record,
        "model_path": model_path,
        "a_run": {"dir": record["a_dir"], "config": record.get("a_config", {})},
        "c_run": {"dir": record["c_dir"], "config": record.get("c_config", {})},
        "b_cfg": make_b_config(record["b_config"]),
        "score": float(score),
        "combo_key": record.get("combo_key", "best"),
    }


def load_struct_candidates_from_dir(out_dir, args):
    with open(osp.join(out_dir, "summary.json"), "r") as f:
        summary = json.load(f)
    cfg = load_config(out_dir)
    metric = effective_struct_metric(args)
    records = {}

    def add_record(record, model_path=None):
        combo_key = record.get("combo_key", "best")
        if combo_key in records:
            if model_path and not records[combo_key].get("model_path"):
                records[combo_key]["model_path"] = model_path
            return
        records[combo_key] = {"record": record, "model_path": model_path}

    best = summary.get("best", {})
    if best:
        add_record(best, resolve_model_path(out_dir, best.get("model_path")))
    for record in summary.get("top_by_validation", []):
        add_record(record)

    cache_root = osp.join(out_dir, "combo_cache")
    for record_path in glob.glob(osp.join(cache_root, "combo-*", "record.json")):
        try:
            with open(record_path, "r") as f:
                payload = json.load(f)
            if payload.get("format") != "abc_lgbm_combo_v1":
                continue
            model_path = payload.get("model_path")
            if not model_path or not osp.isfile(model_path):
                model_path = osp.join(osp.dirname(record_path), "model.txt")
            if not osp.isfile(model_path):
                continue
            add_record(payload["record"], model_path)
        except Exception as exc:
            print(f"[hybrid] skip struct combo cache {record_path}: {exc}", flush=True)

    candidates = []
    for item in records.values():
        record = item["record"]
        model_path = item.get("model_path")
        combo_key = record.get("combo_key")
        if not model_path and combo_key:
            candidate_model = osp.join(combo_cache_dir(out_dir, combo_key), "model.txt")
            if osp.isfile(candidate_model):
                model_path = candidate_model
        if not model_path or not osp.isfile(model_path):
            continue
        score = metric_value(record["val_metrics"], metric)
        candidates.append(make_struct_candidate(out_dir, cfg, summary, record, model_path, score))

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if args.struct_combo_key:
        candidates = [item for item in candidates if str(item.get("combo_key")) == str(args.struct_combo_key)]
    return candidates[: int(args.top_k_struct)]


def load_time_run(out_dir):
    return {
        "dir": out_dir,
        "config": load_config(out_dir),
        "metrics": load_metrics(out_dir),
    }


def matches_common_config(cfg, args):
    return (
        str(cfg.get("dataset")) == str(args.dataset)
        and int(cfg.get("seed", -1)) == int(args.seed)
        and int(cfg.get("ns_q", 10**9)) == int(args.ns_q)
        and int(cfg.get("ns_seed", 10**9)) == int(args.ns_seed)
        and abs(float(cfg.get("train_predict_ratio", -1.0)) - float(args.train_predict_ratio)) <= 1e-12
    )


def find_struct_runs(args):
    if not args.struct_dir:
        raise SystemExit("train_all.py passes a single explicit struct_dir; none was provided")
    cfg = load_config(args.struct_dir)
    if not matches_common_config(cfg, args):
        raise SystemExit(f"Struct run config does not match hybrid args: {args.struct_dir}")
    runs = load_struct_candidates_from_dir(args.struct_dir, args)
    if not runs:
        raise SystemExit(f"No usable Struct combo cache found in {args.struct_dir}")
    return runs


def find_time_runs(args):
    if not args.time_dir:
        raise SystemExit("train_all.py passes a single explicit time_dir; none was provided")
    required_modes = ("train", "val")
    if getattr(args, "eval_test", True):
        required_modes = required_modes + ("test",)
    if not is_run_complete(args.time_dir, modes=required_modes):
        raise SystemExit(f"Time run is missing train/val/test score files: {args.time_dir}")
    run = load_time_run(args.time_dir)
    if not matches_common_config(run["config"], args):
        raise SystemExit(f"Time run config does not match hybrid args: {args.time_dir}")
    run["score"] = score_from_metrics(run["metrics"], effective_time_metric(args))
    return [run]


def minmax_by_query(scores, valid):
    low = np.min(np.where(valid, scores, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(valid, scores, -np.inf), axis=1, keepdims=True)
    denom = np.maximum(high - low, EPS)
    out = (scores - low) / denom
    return np.where(valid, out, 0.0).astype(np.float32, copy=False)


def zscore_by_query(scores, valid):
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(valid, scores, 0.0), axis=1, keepdims=True) / count
    var = np.sum(np.where(valid, (scores - mean) ** 2, 0.0), axis=1, keepdims=True) / count
    return np.where(valid, (scores - mean) / np.sqrt(np.maximum(var, EPS)), 0.0).astype(np.float32, copy=False)


class HybridFeatureBuilder:
    def __init__(self):
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
        ):
            self._add(name)

    def make(self, struct_scores, time_scores, base_scores, valid):
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
        return np.stack(features, axis=2).astype(np.float32, copy=False)


def predict_struct_scores(struct_model, cube, valid):
    flat_valid = valid.reshape(-1)
    pred = np.zeros(flat_valid.shape[0], dtype=np.float32)
    if np.any(flat_valid):
        flat_cube = cube.reshape(-1, cube.shape[-1])
        pred[flat_valid] = predict_lgbm(struct_model, flat_cube[flat_valid])
    return pred.reshape(valid.shape)


def make_base_scores(struct_scores, time_scores, valid):
    return ((minmax_by_query(struct_scores, valid) + minmax_by_query(time_scores, valid)) * 0.5).astype(np.float32, copy=False)


def topk_mask(ranks, valid, topk, extra=0):
    if int(topk) <= 0:
        return valid.copy()
    return (ranks <= int(topk) + int(extra)) & valid


def train_selection(struct_ranks, time_ranks, base_ranks, valid, topk):
    selected = (
        topk_mask(struct_ranks, valid, topk, extra=1)
        | topk_mask(time_ranks, valid, topk, extra=1)
        | topk_mask(base_ranks, valid, topk, extra=1)
    )
    selected[:, 0] = True
    return selected & valid


def iter_hybrid_blocks(context, split, args):
    time_store = ScoreStore(context.time_dir, split)
    row_offset = 0
    iterator = iter_feature_blocks(
        context.struct["a_run"],
        context.struct["c_run"],
        context.struct["b_cfg"],
        context.data,
        split,
        context.struct_feature_builder,
        args,
    )

    for batch_data, struct_valid, _, _, _, _, cube in iterator:
        width = struct_valid.shape[1] - 1
        end = row_offset + len(batch_data)
        time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
        if time_neg.shape[1] != width:
            raise ValueError(f"time score width mismatch at {split}: struct={width}, time={time_neg.shape[1]}")

        time_valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), time_mask), axis=1)
        valid = struct_valid & time_valid
        struct_scores = predict_struct_scores(context.struct["model"], cube, valid)
        time_scores = np.concatenate((time_pos, time_neg), axis=1).astype(np.float32, copy=False)
        time_scores = np.where(valid, time_scores, 0.0).astype(np.float32, copy=False)
        base_scores = make_base_scores(struct_scores, time_scores, valid)
        features = context.hybrid_feature_builder.make(struct_scores, time_scores, base_scores, valid)
        yield valid, struct_scores, time_scores, base_scores, features
        row_offset = end

    if row_offset != time_store.num_rows:
        raise ValueError(f"time row count mismatch for {split}: stream={row_offset}, time={time_store.num_rows}")


def build_training_matrix(context, args):
    return build_hybrid_ranker_matrix(context, "train", args.train_topk, args)


def build_validation_matrix(context, args):
    raise RuntimeError("validation metrics are evaluated in streaming mode; do not build a full validation matrix")


def eval_stream_args(args):
    values = vars(args).copy()
    values["block_size"] = int(getattr(args, "eval_batch_size", getattr(args, "block_size", 128)))
    return SimpleNamespace(**values)


def split_query_count(data, split):
    key = "train_list" if split == "train" else f"{split}_list"
    snapshots = data[key]
    if split == "train":
        snapshots = snapshots[data["train_predict_start_idx"] :]
    return int(sum(len(events) for events, _, _ in snapshots))


def build_hybrid_ranker_matrix(context, split, topk, args):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0

    iterator = iter_hybrid_blocks(context, split, args)
    for valid, struct_scores, time_scores, base_scores, features in tqdm(iterator, desc=f"hybrid_{split}_matrix", leave=False):
        if topk is None:
            selected = valid.copy()
        else:
            selected = train_selection(
                dense_rank(struct_scores, valid),
                dense_rank(time_scores, valid),
                dense_rank(base_scores, valid),
                valid,
                topk,
            )
        for row in range(selected.shape[0]):
            cols = np.flatnonzero(selected[row])
            if len(cols) <= 1:
                continue
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[np.flatnonzero(cols == 0)[0]] = 1.0
            X_parts.append(features[row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)

    if not X_parts:
        raise ValueError(f"no hybrid {split} rows built; check train_predict_ratio/topk")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {"queries": int(queries), "rows": int(rows)},
    )


def evaluate_hybrid(context, model, split, args):
    sums = {}
    stream_args = eval_stream_args(args)
    iterator = iter_hybrid_blocks(context, split, stream_args)
    for valid, struct_scores, time_scores, base_scores, features in tqdm(iterator, desc=f"hybrid_{split}", leave=False):
        loose_ranks = []
        strict_ranks = []
        avg_ranks = []
        X_parts = []
        slices = []
        cursor = 0

        for row in range(valid.shape[0]):
            cols = np.flatnonzero(valid[row])
            start = cursor
            X_parts.append(features[row, cols, :])
            cursor += len(cols)
            slices.append((start, cursor, 0))

        if X_parts:
            X = np.vstack(X_parts).astype(np.float32, copy=False)
            pred = model.predict(X).astype(np.float32, copy=False)
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

    from utils import finalize_metric_sums

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


def fit_hybrid_lgbm(X, y, group, feature_builder, args, params=None, eval_data=None):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lgbm_hybrid.py requires lightgbm") from exc

    params = default_lgbm_params(args) if params is None else params
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
        random_state=args.seed,
        n_jobs=args.num_threads,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    fit_kwargs = {"group": group.tolist(), "feature_name": feature_builder.feature_names}
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


def tune_hybrid_lgbm(context, X_train, y_train, group, args):
    n_trials = int(getattr(args, "lgbm_n_trials", 30))
    if n_trials <= 0:
        params = default_lgbm_params(args)
        model = fit_hybrid_lgbm(
            X_train,
            y_train,
            group,
            context.hybrid_feature_builder,
            args,
            params=params,
            eval_data=None,
        )
        val_metrics = evaluate_hybrid(context, model, "val", args)
        score = metric_value(val_metrics, args.metric)
        best_iteration = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
        print(
            f"[hybrid] fixed LGBM val_{ranking_metric_key(args.metric)}={score:.5f} "
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
        raise RuntimeError("Optuna is required for validation-based hybrid LGBM parameter selection") from exc

    best = {"score": -float("inf"), "model": None, "params": None, "val_metrics": None}
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = sample_lgbm_params(trial)
        model = fit_hybrid_lgbm(
            X_train,
            y_train,
            group,
            context.hybrid_feature_builder,
            args,
            params=params,
            eval_data=None,
        )
        val_metrics = evaluate_hybrid(context, model, "val", args)
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
        raise RuntimeError("Hybrid LGBM tuning failed to produce a model")
    print(
        f"[hybrid] best LGBM trial={study.best_trial.number} "
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


def save_lgbm_model(model, path):
    booster = getattr(model, "booster_", model)
    booster.save_model(path)


def pair_key(args, struct, time_run):
    return stable_hash(
        {
            "args": {
                "dataset": args.dataset,
                "seed": args.seed,
                "ns_q": args.ns_q,
                "ns_seed": args.ns_seed,
                "train_predict_ratio": args.train_predict_ratio,
                "metric": args.metric,
                "train_topk": args.train_topk,
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
            "struct_dir": struct["dir"],
            "struct_combo_key": struct.get("combo_key"),
            "struct_model_path": struct["model_path"],
            "time_dir": time_run["dir"],
        },
        length=16,
    )


def hybrid_cache_dir(out_dir, key):
    return osp.join(out_dir, "pair_cache", f"pair-{key}")


def load_hybrid_cache(out_dir, key):
    cache_dir = hybrid_cache_dir(out_dir, key)
    record_path = osp.join(cache_dir, "record.json")
    model_path = osp.join(cache_dir, "model.txt")
    if not osp.isfile(record_path) or not osp.isfile(model_path):
        return None
    try:
        with open(record_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[hybrid] skip pair cache {cache_dir}: {exc}", flush=True)
        return None
    if payload.get("format") != "hybrid_lgbm_pair_v1" or payload.get("pair_key") != key:
        return None
    return payload["record"], model_path


def save_hybrid_cache(out_dir, key, record, model):
    cache_dir = hybrid_cache_dir(out_dir, key)
    os.makedirs(cache_dir, exist_ok=True)
    model_path = osp.join(cache_dir, "model.txt")
    save_lgbm_model(model, model_path)
    payload = {
        "format": "hybrid_lgbm_pair_v1",
        "pair_key": key,
        "record": record,
        "model_path": model_path,
    }
    with open(osp.join(cache_dir, "record.json"), "w") as f:
        json.dump(payload, f, indent=2)


def make_out_dir(args, struct_runs, time_runs):
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
    cfg_hash = stable_hash(
        {
            "args": vars(args),
            "struct_candidates": [
                {
                    "dir": item["dir"],
                    "combo_key": item.get("combo_key"),
                    "model_path": item["model_path"],
                }
                for item in struct_runs
            ],
            "time_candidates": [item["dir"] for item in time_runs],
        },
        length=12,
    )
    return osp.join(
        args.out_prefix,
        args.dataset,
        f"seed{args.seed}",
        f"tr{args.train_predict_ratio:g}_nq{args.ns_q}_ns{args.ns_seed}",
        f"m-{safe_token(args.metric)}_s{args.top_k_struct}_t{args.top_k_time}"
        f"_traink{args.train_topk}_evalfull",
        f"lgbm-{lgbm_hash}",
        f"cfg-{cfg_hash}",
    )


def validate_args(args):
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if not 0.0 < float(args.train_predict_ratio) < 1.0:
        raise ValueError("--train_predict_ratio must be in (0, 1)")
    if args.train_topk <= 0:
        raise ValueError("--train_topk must be positive")
    if int(getattr(args, "eval_batch_size", getattr(args, "block_size", 128))) <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if args.top_k_struct <= 0 or args.top_k_time <= 0:
        raise ValueError("--top_k_struct/--top_k_time must be positive")
    ranking_metric_key(args.metric, strict=True)
    if args.struct_metric:
        ranking_metric_key(args.struct_metric, strict=True)
    if args.time_metric:
        ranking_metric_key(args.time_metric, strict=True)


def run(args):
    validate_args(args)
    set_random_seed(args.seed)
    start_time = time.time()

    struct_runs = find_struct_runs(args)
    time_runs = find_time_runs(args)
    out_dir = make_out_dir(args, struct_runs, time_runs)
    if osp.exists(osp.join(out_dir, "metrics.json")) and osp.exists(osp.join(out_dir, "best_hybrid_lgbm.txt")) and not args.force:
        cached = load_metrics(out_dir)
        if not getattr(args, "eval_test", True) or cached.get("test_metrics") or cached.get("best", {}).get("test_metrics"):
            print(f"[hybrid] existing result -> {out_dir}", flush=True)
            return cached

    os.makedirs(out_dir, exist_ok=True)
    print(f"[hybrid] output -> {out_dir}", flush=True)
    print(
        f"[hybrid] struct candidates={len(struct_runs)} "
        f"metric={ranking_metric_key(effective_struct_metric(args))}",
        flush=True,
    )
    for idx, item in enumerate(struct_runs, start=1):
        print(
            f"[hybrid]   S{idx}: score={item['score']:.5f} "
            f"combo={item.get('combo_key')} dir={item['dir']}",
            flush=True,
        )
    print(
        f"[hybrid] time candidates={len(time_runs)} "
        f"metric={ranking_metric_key(effective_time_metric(args))}",
        flush=True,
    )
    for idx, item in enumerate(time_runs, start=1):
        print(f"[hybrid]   T{idx}: score={item['score']:.5f} dir={item['dir']}", flush=True)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    if not data["train_predict_count"]:
        raise ValueError("train_predict_ratio selected no training timestamps")

    struct_feature_builder = FeatureBuilder(data["num_rels"])
    hybrid_feature_builder = HybridFeatureBuilder()
    for struct in struct_runs:
        expected_names = struct["summary"].get("feature_names")
        if expected_names and expected_names != struct_feature_builder.feature_names:
            raise ValueError(
                f"structure-combine feature protocol mismatch in {struct['dir']}; "
                "rerun the structure-combine stage with the current code"
            )

    candidates = []
    best = None
    pairs = [(struct, time_run) for struct in struct_runs for time_run in time_runs]
    for idx, (struct, time_run) in enumerate(pairs, start=1):
        key = pair_key(args, struct, time_run)
        print(
            f"\n[hybrid] pair {idx}/{len(pairs)} "
            f"S={struct.get('combo_key')} T={osp.basename(time_run['dir'])}",
            flush=True,
        )
        cached = load_hybrid_cache(out_dir, key)
        if cached is not None:
            record, model_path = cached
            candidates.append(record)
            print(f"[hybrid] cache hit pair={key}", flush=True)
            print(f"[hybrid] val {format_metrics(record['val_metrics'])}", flush=True)
            if best is None or record["val_score"] > best["record"]["val_score"]:
                if best is not None:
                    del best["model"]
                    gc.collect()
                best = {
                    "record": record,
                    "model": load_lgbm_model(model_path),
                    "struct": struct,
                    "time": time_run,
                }
            continue

        struct_with_model = dict(struct)
        struct_with_model["model"] = load_lgbm_model(struct["model_path"])
        context = SimpleNamespace(
            data=data,
            struct=struct_with_model,
            time_dir=time_run["dir"],
            struct_feature_builder=struct_feature_builder,
            hybrid_feature_builder=hybrid_feature_builder,
        )
        X_train, y_train, group, train_info = build_training_matrix(context, args)
        print(
            f"[hybrid] train rows={train_info['rows']} queries={train_info['queries']} "
            f"features={X_train.shape[1]}",
            flush=True,
        )
        val_info = {
            "mode": "full_metric_stream",
            "queries": split_query_count(data, "val"),
            "eval_batch_size": int(getattr(args, "eval_batch_size", getattr(args, "block_size", 128))),
        }
        print(
            f"[hybrid] val metric stream queries={val_info['queries']} "
            f"eval_batch_size={val_info['eval_batch_size']}",
            flush=True,
        )
        model, lgbm_tuning = tune_hybrid_lgbm(
            context,
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
            "pair_key": key,
            "val_score": float(val_score),
            "val_metrics": val_metrics,
            "train_info": train_info,
            "val_eval_info": val_info,
            "val_tune_info": val_info,
            "struct": {
                "dir": struct["dir"],
                "combo_key": struct.get("combo_key"),
                "model_path": struct["model_path"],
                "score": float(struct["score"]),
                "record": struct["record"],
            },
            "time": {
                "dir": time_run["dir"],
                "score": float(time_run["score"]),
                "metrics": time_run["metrics"],
            },
            "lgbm_tuning": lgbm_tuning,
        }
        candidates.append(record)
        print(f"[hybrid] val {format_metrics(val_metrics)}", flush=True)
        save_hybrid_cache(out_dir, key, record, model)

        if best is None or val_score > best["record"]["val_score"]:
            if best is not None:
                del best["model"]
                gc.collect()
            best = {"record": record, "model": model, "struct": struct, "time": time_run}
        else:
            del model
            gc.collect()
        del struct_with_model["model"]
        gc.collect()

    candidates.sort(key=lambda item: item["val_score"], reverse=True)
    best_record = best["record"]
    best_struct = dict(best["struct"])
    best_struct["model"] = load_lgbm_model(best_struct["model_path"])
    best_context = SimpleNamespace(
        data=data,
        struct=best_struct,
        time_dir=best["time"]["dir"],
        struct_feature_builder=struct_feature_builder,
        hybrid_feature_builder=hybrid_feature_builder,
    )
    train_metrics = evaluate_hybrid(best_context, best["model"], "train", args)
    test_metrics = None
    if getattr(args, "eval_test", True):
        test_metrics = evaluate_hybrid(best_context, best["model"], "test", args)
    val_metrics = best_record["val_metrics"]
    val_score = float(best_record["val_score"])
    test_score = metric_value(test_metrics, args.metric) if test_metrics is not None else 0.0

    print(f"\n[hybrid] best val {ranking_metric_key(args.metric)}={val_score:.5f}", flush=True)
    print(f"[hybrid] best struct: {best_record['struct']['combo_key']} {best_record['struct']['dir']}", flush=True)
    print(f"[hybrid] best time: {best_record['time']['dir']}", flush=True)
    print(f"[hybrid] train {format_metrics(train_metrics)}", flush=True)
    print(f"[hybrid] val   {format_metrics(val_metrics)}", flush=True)
    if test_metrics is not None:
        print(f"[hybrid] test  {format_metrics(test_metrics)}", flush=True)
    print(
        f"[hybrid] selected val {ranking_metric_key(args.metric)}={val_score:.5f} "
        f"test={test_score:.5f}" if test_metrics is not None else
        f"[hybrid] selected val {ranking_metric_key(args.metric)}={val_score:.5f}",
        flush=True,
    )

    model_path = osp.join(out_dir, "best_hybrid_lgbm.txt")
    save_lgbm_model(best["model"], model_path)
    config = vars(args).copy()
    config.update(
        {
            "out_dir": out_dir,
            "struct_candidates": [
                {
                    "dir": item["dir"],
                    "combo_key": item.get("combo_key"),
                    "model_path": item["model_path"],
                    "score": float(item["score"]),
                }
                for item in struct_runs
            ],
            "time_candidates": [
                {"dir": item["dir"], "score": float(item["score"])} for item in time_runs
            ],
            "feature_names": hybrid_feature_builder.feature_names,
        }
    )
    summary = {
        "format": "hybrid_lgbm_v2",
        "dataset": args.dataset,
        "selection_metric": ranking_metric_key(args.metric),
        "struct_preselect_metric": ranking_metric_key(effective_struct_metric(args)),
        "time_preselect_metric": ranking_metric_key(effective_time_metric(args)),
        "train_info": best_record["train_info"],
        "feature_names": hybrid_feature_builder.feature_names,
        "model_path": model_path,
        "best": {
            **best_record,
            "train_metrics": train_metrics,
            "model_path": model_path,
        },
        "top_by_validation": candidates,
        "val_score": float(val_score),
        "test_score": float(test_score),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "runtime_sec": float(time.time() - start_time),
    }
    if test_metrics is not None:
        summary["best"]["test_metrics"] = test_metrics
        summary["test_metrics"] = test_metrics
    save_config(out_dir, config)
    save_metrics(out_dir, summary)
    with open(osp.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[hybrid] saved -> {out_dir}", flush=True)
    return summary
