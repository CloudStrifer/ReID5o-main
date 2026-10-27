"""Frozen-checkpoint missing-rate evaluation; exports masks and modality weights."""

import argparse
import json
from pathlib import Path
from utils.iotools import load_train_configs
from utils.options import get_args
from utils.checkpoint import load_for_inference
from utils.metrics import evaluator_from_loaders
from datasets import build_dataloader
from train import resolve_device, seed_everything, configure_runtime
from utils.provenance import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument(
        "--missing_rates", nargs="+", type=float, default=[0.0, 0.2, 0.3, 0.5]
    )
    p.add_argument("--gallery_missing_rate", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--gallery_file", default=None)
    p.add_argument("--query_tasks", nargs="*", default=[])
    cli = p.parse_args()
    args = get_args([])
    vars(args).update(dict(load_train_configs(cli.config_file)))
    args.training, args.device, args.eval_seed = False, cli.device, cli.eval_seed
    args.eval_gallery_missing_rate = cli.gallery_missing_rate
    configure_runtime(args)
    if cli.gallery_file:
        args.gallery_file = cli.gallery_file
    checkpoint = cli.checkpoint or str(Path(args.output_dir) / "best.pth")
    seed_everything(args.seed)
    model = load_for_inference(args, checkpoint, resolve_device(args.device))
    checkpoint_hash = sha256_file(checkpoint)
    loaders = build_dataloader(args)
    if loaders[-1] != model.num_classes:
        raise ValueError("Dataset train class count differs from checkpoint")
    for rate in cli.missing_rates:
        args.eval_missing_rate = rate
        args.protocol_dir = str(
            Path(args.output_dir)
            / "evaluation"
            / f"missing_{rate:g}_seed_{args.eval_seed}"
        )
        evaluator = evaluator_from_loaders(loaders[:-1], args)
        if cli.query_tasks:
            unknown = set(cli.query_tasks) - set(evaluator.query_loaders)
            if unknown:
                raise ValueError(f"Unknown query tasks: {unknown}")
            evaluator.query_loaders = {
                k: v for k, v in evaluator.query_loaders.items() if k in cli.query_tasks
            }
        mean = evaluator.eval(model)
        output = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "config": vars(args),
            "missing_rate": rate,
            "gallery_missing_rate": args.eval_gallery_missing_rate,
            "seed": args.eval_seed,
            "mean_task_R1": mean,
            "results": evaluator.results,
        }
        path = Path(args.protocol_dir) / "metrics.json"
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
