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


def add_bool_arg(parser, name, default=False, help_text=""):
    flag = name.replace("_", "-")
    opts = [f"--{name}"]
    if flag != name:
        opts.append(f"--{flag}")
    parser.add_argument(*opts, dest=name, action="store_true", help=help_text)
    no_opts = [f"--no_{name}", f"--no-{flag}"]
    parser.add_argument(*no_opts, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


def add_auto_bool_arg(parser, name, help_text=""):
    flag = name.replace("_", "-")
    opts = [f"--{name}"]
    if flag != name:
        opts.append(f"--{flag}")
    parser.add_argument(*opts, dest=name, action="store_true", help=help_text)
    no_opts = [f"--no_{name}", f"--no-{flag}"]
    parser.add_argument(*no_opts, dest=name, action="store_false")
    parser.set_defaults(**{name: None})


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
    add_bool_arg(parser, "force", default=False)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_neg_chunk", type=int, default=512)
    parser.add_argument("--max_eval_pairs", type=int, default=500000)
    parser.add_argument("--eval_node_preload_chunk", type=int, default=65536)
    parser.add_argument("--max_eval_node_cache_mb", type=float, default=4096.0)
    add_bool_arg(parser, "preload_eval_nodes", default=True)
    add_bool_arg(parser, "dense_eval_node_cache", default=True)
    add_bool_arg(parser, "cache_eval_source", default=False)
    add_bool_arg(parser, "eval_test", default=True)

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
    add_bool_arg(parser, "curriculum_raw_age", default=False)

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
    add_bool_arg(parser, "use_single_layer", default=False)
    parser.add_argument("--predictor_mode", choices=("diag", "concat"), default="diag")
    parser.add_argument("--event_encoder", choices=("mixer", "transformer"), default="mixer")
    parser.add_argument("--transformer_heads", type=int, default=2)
    parser.add_argument("--transformer_ff_dim", type=int, default=None)
    add_bool_arg(parser, "use_cross_history", default=False)
    parser.add_argument("--cross_heads", type=int, default=2)

    add_bool_arg(parser, "use_neighbor_id", default=True)
    add_bool_arg(parser, "use_abs_time", default=True)
    parser.add_argument("--abs_time_periods", default="7,30,180,365")
    parser.add_argument("--abs_time_harmonics", type=int, default=1)
    add_bool_arg(parser, "abs_time_use_raw", default=False)
    add_bool_arg(parser, "use_query_gate", default=True)
    parser.add_argument("--query_gate_type", choices=("channel", "scalar"), default="channel")
    add_bool_arg(parser, "use_rank_pos", default=True)

    add_auto_bool_arg(
        parser,
        "use_node_geo",
        help_text="Use Yelp/THG node geo features. Default: on for Yelp datasets, off otherwise.",
    )
    add_auto_bool_arg(
        parser,
        "thg_time_days",
        help_text="Use raw timestamp days as model time. Default: on for Yelp datasets, off otherwise.",
    )
    parser.add_argument("--user_center_half_life_days", type=float, default=365.0)

    parser.add_argument(
        "--no_retrain_on_train_prefix",
        "--no-retrain-on-train-prefix",
        action="store_true",
        default=False,
    )
    add_bool_arg(parser, "reuse_no_retrain_full", default=True)
    add_bool_arg(parser, "profile_sync", default=False)
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
