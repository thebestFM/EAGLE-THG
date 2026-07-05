import copy
import json
import os
import os.path as osp
from types import SimpleNamespace

import train_new_hybrid_save_top10 as hybrid


ABLATION_GROUPS = ("recurrence", "direct", "shared", "time", "cross_signal")
ABLATION_ALIASES = {
    "wo_recurrence": "recurrence",
    "without_recurrence": "recurrence",
    "w/o_recurrence": "recurrence",
    "wo_direct": "direct",
    "without_direct": "direct",
    "w/o_direct": "direct",
    "wo_shared": "shared",
    "without_shared": "shared",
    "w/o_shared": "shared",
    "wo_time": "time",
    "without_time": "time",
    "w/o_time": "time",
    "wo_cross": "cross_signal",
    "wo_cross_signal": "cross_signal",
    "without_cross": "cross_signal",
    "without_cross_signal": "cross_signal",
    "w/o_cross": "cross_signal",
    "w/o_cross_signal": "cross_signal",
}


def normalize_ablation_name(name):
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = ABLATION_ALIASES.get(key, key)
    if key == "all":
        return key
    if key not in ABLATION_GROUPS:
        raise ValueError(
            f"unknown --ablation {name!r}; expected one of {','.join(ABLATION_GROUPS)},all"
        )
    return key


def output_dir_from_summary(summary):
    model_path = summary.get("best", {}).get("model_path", "")
    if model_path:
        return osp.dirname(osp.dirname(model_path))
    return ""


def compact_result(group, summary):
    best = summary["best"]
    return {
        "ablation": group,
        "out_dir": output_dir_from_summary(summary),
        "pair_id": best.get("pair_id", ""),
        "test_metrics": best.get("test_metrics", {}),
        "selection_metrics": best.get("selection_metrics", {}),
        "rescue_topk": best.get("rescue_topk"),
        "preset": best.get("preset"),
        "removed_feature_names": best.get("removed_feature_names", []),
        "kept_feature_count": best.get("kept_feature_count"),
        "full_feature_count": best.get("full_feature_count"),
    }


def run_one(base_args, group):
    args = SimpleNamespace(**copy.deepcopy(vars(base_args)))
    args.ablation_feature_group = group
    args.output_root = osp.join(str(base_args.output_root), f"wo_{group}")
    print(
        f"[HybridAblation] running w/o {group}: output_root={args.output_root}",
        flush=True,
    )
    summary = hybrid.run(args)
    result = compact_result(group, summary)
    metrics = result["test_metrics"]
    print(
        f"[HybridAblation] done w/o {group}: "
        f"MRR={metrics.get('mrr_strict', metrics.get('mrr', 0.0)):.5f} "
        f"HR@1={metrics.get('hit@1_strict', metrics.get('hr1', 0.0)):.5f} "
        f"HR@10={metrics.get('hit@10_strict', metrics.get('hr10', 0.0)):.5f} "
        f"kept={result['kept_feature_count']}/{result['full_feature_count']}",
        flush=True,
    )
    return result


def save_ablation_summary(args, results):
    root = osp.join(str(args.output_root), str(args.dataset), f"seed{int(args.seed)}")
    os.makedirs(root, exist_ok=True)
    path = osp.join(root, "ablation_summary.json")
    payload = {
        "format": "hybrid_ablation_summary_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "requested_ablation": args.ablation,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[HybridAblation] summary={path}", flush=True)
    return path


def run(args):
    requested = normalize_ablation_name(args.ablation)
    groups = list(ABLATION_GROUPS) if requested == "all" else [requested]
    results = [run_one(args, group) for group in groups]
    save_ablation_summary(args, results)
    return results


def parse_args():
    parser = hybrid.build_arg_parser(
        "Run source-level ablations for the rich rescue hybrid reranker."
    )
    parser.set_defaults(output_root="results_hybrid_ablation")
    parser.add_argument(
        "--ablation",
        default="all",
        help=(
            "Ablation to run: recurrence,direct,shared,time,cross_signal,all. "
            "Aliases such as wo_direct are also accepted."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
