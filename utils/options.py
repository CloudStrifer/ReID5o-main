import argparse
from pathlib import Path
import yaml


def get_args(argv=None):
    p = argparse.ArgumentParser(
        description="MFCRAF paper-aligned training and evaluation"
    )
    p.add_argument("--config_file", default=None)

    def flag(name, default=True, help=""):
        p.add_argument(
            "--" + name,
            "--" + name.replace("_", "-"),
            dest=name,
            default=default,
            action=argparse.BooleanOptionalAction,
            help=help,
        )

    p.add_argument("--name", default="MFCRAF")
    p.add_argument("--output_dir", default="logs")
    p.add_argument("--root_dir", default="./data")
    p.add_argument("--dataset_name", default="ORBench")
    p.add_argument(
        "--split_file",
        default=None,
        help="Official CUHK path-to-split JSON or reid_raw.json",
    )
    p.add_argument(
        "--gallery_file",
        default=None,
        help="Optional multimodal test-gallery manifest JSON",
    )
    p.add_argument("--missing_sketch_policy", choices=["mask", "error"], default="mask")
    p.add_argument("--val_dataset", choices=["val", "none"], default="val")
    p.add_argument("--pretrain_choice", default="ViT-B/16")
    p.add_argument(
        "--img_size", type=int, nargs=2, default=[384, 128], metavar=("HEIGHT", "WIDTH")
    )
    p.add_argument("--stride_size", type=int, default=16)
    p.add_argument("--text_length", type=int, default=77)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--loss_names", default="mm_sdm+id")
    p.add_argument("--id_loss_weight", type=float, default=1.0)
    for name, default in [
        ("add_lora", True),
        ("freeze_vision_encoder", True),
        ("freeze_text_encoder", False),
        ("use_missing_aware", True),
        ("use_cross_modal_completion", True),
        ("use_reliability_fusion", True),
        ("use_completion_inference", True),
        ("use_reliability_fusion_inference", True),
        ("use_quality_indicators", True),
        ("img_aug", False),
        ("compute_mAP", True),
        ("resume", False),
        ("amp", False),
    ]:
        flag(name, default)
    flag(
        "disable_mkldnn",
        False,
        "Disable oneDNN on CPUs that cannot run its optimized kernels",
    )
    p.add_argument("--cpu_threads", type=int, default=0)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument(
        "--spa_experts", "--num_loras", dest="spa_experts", type=int, default=4
    )
    p.add_argument("--spa_parts", type=int, default=3)
    p.add_argument("--lora_layers", type=int, default=2)
    p.add_argument("--modality_dropout_prob", type=float, default=0.3)
    p.add_argument("--keep_rgb_prob", type=float, default=0.9)
    p.add_argument("--min_modalities_keep", type=int, default=1)
    p.add_argument("--consistency_loss_weight", type=float, default=0.1)
    p.add_argument("--completion_num_heads", type=int, default=8)
    p.add_argument("--completion_num_layers", type=int, default=2)
    p.add_argument("--completion_dropout", type=float, default=0.1)
    p.add_argument("--completion_recon_loss_weight", type=float, default=0.3)
    p.add_argument("--completion_cycle_loss_weight", type=float, default=0.1)
    p.add_argument("--cycle_pairs_per_step", type=int, default=3)
    p.add_argument("--reliability_hidden_dim", type=int, default=256)
    p.add_argument("--fusion_temperature", type=float, default=1.0)
    p.add_argument("--fusion_entropy_target", type=float, default=0.3)
    p.add_argument("--fusion_uncertainty_margin", type=float, default=0.5)
    p.add_argument("--fusion_sparsity_weight", type=float, default=0.01)
    p.add_argument("--fusion_uncertainty_weight", type=float, default=0.02)
    for name, default in [
        ("modality_dropout_warmup_epochs", 5),
        ("completion_warmup_epochs", 3),
        ("fusion_warmup_epochs", 5),
        ("num_epoch", 30),
        ("warmup_epochs", 5),
        ("batch_size", 8),
        ("test_batch_size", 64),
        ("num_workers", 0),
        ("eval_period", 2),
        ("log_period", 100),
        ("seed", 42),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--optimizer", choices=["Adam", "AdamW", "SGD"], default="Adam")
    for name, default in [
        ("lr", 1e-5),
        ("bias_lr_factor", 2.0),
        ("lora_lr_factor", 5.0),
        ("momentum", 0.9),
        ("weight_decay", 4e-5),
        ("weight_decay_bias", 0.0),
        ("alpha", 0.9),
        ("beta", 0.999),
        ("warmup_factor", 0.1),
        ("gamma", 0.1),
        ("target_lr", 0.0),
        ("power", 0.9),
    ]:
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--warmup_method", default="linear")
    p.add_argument("--lrscheduler", default="cosine")
    p.add_argument("--milestones", type=int, nargs="+", default=[20, 50])
    p.add_argument("--resume_ckpt_file", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--test", dest="training", action="store_false", default=True)
    p.add_argument("--eval_missing_rate", type=float, default=0.0)
    p.add_argument("--eval_gallery_missing_rate", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--metric_chunk_size", type=int, default=128)
    p.add_argument("--protocol_dir", default="")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_file", default=None)
    config, _ = pre.parse_known_args(argv)
    if config.config_file:
        values = yaml.safe_load(
            Path(config.config_file).read_text(encoding="utf-8-sig")
        )
        if not isinstance(values, dict):
            p.error("Config must be a YAML mapping")
        known = {a.dest for a in p._actions} | {
            "num_classes",
            "architecture_version",
            "checkpoint_selection",
            "dataset_manifest",
        }
        unknown = set(values) - known
        if unknown:
            p.error(f"Unknown configuration keys: {sorted(unknown)}")
        p.set_defaults(**values)
    args = p.parse_args(argv)
    if args.temperature <= 0:
        p.error("--temperature must be positive")
    if args.metric_chunk_size < 1:
        p.error("--metric_chunk_size must be positive")
    if args.cycle_pairs_per_step < 1:
        p.error("--cycle_pairs_per_step must be positive")
    if (
        min(args.eval_period, args.log_period, args.batch_size, args.test_batch_size)
        < 1
    ):
        p.error("Logging/evaluation periods and batch sizes must be positive")
    if (
        not 0 <= args.eval_missing_rate <= 1
        or not 0 <= args.eval_gallery_missing_rate <= 1
    ):
        p.error("Missing rates must be in [0,1]")
    if args.num_epoch <= args.warmup_epochs:
        p.error("num_epoch must exceed warmup_epochs")
    return args
