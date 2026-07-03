import argparse
import os.path as osp


THG_DATASETS = {
    "Yelp-NOLA",
    "Yelp-PHL",
    "Yelp-TPA",
    "Yelp-BOI",
    "Yelp-STL",
    "Yelp-SBA",
    "Yelp-RNO",
    "Yelp-IND",
    "Yelp-TUS",
    "Yelp-BNA",
}


def normalize_args(args):
    is_thg = args.dataset in THG_DATASETS
    if args.use_node_geo is None:
        args.use_node_geo = bool(is_thg)
    if args.thg_time_days is None:
        args.thg_time_days = bool(is_thg)
    return args


def ensure_score_files(out_dir, modes):
    missing = []
    for mode in modes:
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
            path = osp.join(out_dir, f"{mode}_{suffix}")
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            f"time score generation incomplete; first missing file: {missing[0]}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate the time module and write train/val/test score stores."
    )

    parser.add_argument("--dataset", default="Yelp-BOI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--force", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_neg_chunk", type=int, default=128)
    parser.add_argument("--max_eval_pairs", type=int, default=125000)
    parser.add_argument("--stream_eval_batch_events", type=int, default=32)
    parser.add_argument("--eval_node_preload_chunk", type=int, default=65536)
    parser.add_argument("--max_eval_node_cache_mb", type=float, default=4096.0)
    parser.set_defaults(preload_eval_nodes=True)
    parser.add_argument("--no_preload_eval_nodes", dest="preload_eval_nodes", action="store_false")
    parser.set_defaults(dense_eval_node_cache=True)
    parser.add_argument("--no_dense_eval_node_cache", dest="dense_eval_node_cache", action="store_false")
    parser.add_argument("--cache_eval_source", action="store_true", default=False)
    parser.set_defaults(eval_test=True)
    parser.add_argument("--no_eval_test", dest="eval_test", action="store_false")

    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--quick_val_events", type=int, default=0)
    parser.add_argument("--quick_val_fraction", type=float, default=0.2)

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--selection_metric", choices=("mrr", "hit10"), default="hit10")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--train_num_neg", type=int, default=8)
    parser.add_argument("--stream_train_batch_events", type=int, default=2048)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument(
        "--train_sampler",
        choices=("exact", "grouped_exact", "fast"),
        default="grouped_exact",
    )
    parser.add_argument("--train_group_matrix_mb", type=float, default=512.0)
    parser.add_argument("--train_loss", choices=("margin", "ce"), default="margin")
    parser.add_argument("--rank_margin", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--curriculum_decay", type=float, default=0.0)
    parser.add_argument("--curriculum_raw_age", action="store_true", default=False)

    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--multi_windows", default="10,40")
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--rel_dim", type=int, default=64)
    parser.add_argument("--node_dim", type=int, default=64)
    parser.add_argument("--event_dim", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--time_min", type=float, default=1.0)
    parser.add_argument("--token_expansion_factor", type=float, default=0.5)
    parser.add_argument("--channel_expansion_factor", type=float, default=4.0)
    parser.add_argument("--use_single_layer", action="store_true", default=False)
    parser.add_argument("--predictor_mode", choices=("diag", "concat"), default="diag")
    parser.add_argument("--event_encoder", choices=("mixer", "transformer"), default="mixer")
    parser.add_argument("--transformer_heads", type=int, default=2)
    parser.add_argument("--transformer_ff_dim", type=int, default=None)
    parser.add_argument("--use_cross_history", action="store_true", default=False)
    parser.add_argument("--cross_heads", type=int, default=2)

    parser.set_defaults(use_neighbor_id=True)
    parser.add_argument("--no_use_neighbor_id", dest="use_neighbor_id", action="store_false")
    parser.set_defaults(use_abs_time=True)
    parser.add_argument("--no_use_abs_time", dest="use_abs_time", action="store_false")
    parser.add_argument("--abs_time_periods", default="7,30,180,365")
    parser.add_argument("--abs_time_harmonics", type=int, default=1)
    parser.add_argument("--abs_time_use_raw", action="store_true", default=False)
    parser.set_defaults(use_query_gate=True)
    parser.add_argument("--no_use_query_gate", dest="use_query_gate", action="store_false")
    parser.add_argument("--query_gate_type", choices=("channel", "scalar"), default="channel")
    parser.set_defaults(use_rank_pos=True)
    parser.add_argument("--no_use_rank_pos", dest="use_rank_pos", action="store_false")

    parser.add_argument("--use_node_geo", dest="use_node_geo", action="store_true", default=None)
    parser.add_argument("--no_use_node_geo", dest="use_node_geo", action="store_false")
    parser.add_argument("--thg_time_days", dest="thg_time_days", action="store_true", default=None)
    parser.add_argument("--no_thg_time_days", dest="thg_time_days", action="store_false")
    parser.add_argument("--user_center_half_life_days", type=float, default=365.0)

    parser.add_argument(
        "--no_retrain_on_train_prefix",
        "--no-retrain-on-train-prefix",
        action="store_true",
        default=False,
    )
    parser.add_argument("--require_existing_best_model", action="store_true", default=False)
    parser.set_defaults(reuse_no_retrain_full=True)
    parser.add_argument("--no_reuse_no_retrain_full", dest="reuse_no_retrain_full", action="store_false")
    parser.add_argument("--profile_sync", action="store_true", default=False)
    parser.set_defaults(use_amp=True)
    parser.add_argument("--no_use_amp", dest="use_amp", action="store_false")
    parser.set_defaults(allow_tf32=True)
    parser.add_argument("--no_allow_tf32", dest="allow_tf32", action="store_false")
    return normalize_args(parser.parse_args())


def main(args):
    from new_single_pipeline import time

    print(
        f"[TrainTime] dataset={args.dataset} output_dir={args.output_dir or '<auto>'}",
        flush=True,
    )
    result = time.main(args)
    out_dir = time.get_out_dir(args)
    modes = ["val"]
    if float(args.train_predict_ratio) > 0.0:
        modes.insert(0, "train")
    if getattr(args, "eval_test", True):
        modes.append("test")
    ensure_score_files(out_dir, modes)
    print(f"[TrainTime] complete result={result} out_dir={out_dir}", flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
