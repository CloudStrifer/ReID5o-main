"""MFCRAF training. Test identities never select the best checkpoint."""

import copy
import json
import logging
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import build_dataloader
from model import build_model
from model.build import ARCHITECTURE_VERSION
from solver import build_optimizer, build_lr_scheduler
from utils.options import get_args
from utils.logger import setup_logger
from utils.iotools import save_train_configs
from utils.checkpoint import Checkpointer
from utils.metrics import evaluator_from_loaders
from utils.provenance import save_run_manifest


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(request):
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if request == "auto"
        else torch.device(request)
    )


def configure_runtime(args):
    if getattr(args, "disable_mkldnn", False):
        torch.backends.mkldnn.enabled = False
    if getattr(args, "cpu_threads", 0) > 0:
        torch.set_num_threads(args.cpu_threads)


def train_one_epoch(
    args,
    model,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    checkpointer=None,
    tensorboard_writer=None,
    scaler=None,
):
    model.train()
    resample = getattr(getattr(train_loader, "dataset", None), "random_sampling", None)
    if resample is not None:
        resample()
    device = next(model.parameters()).device
    scaler = scaler or torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    stats, steps = {}, 0
    for iteration, batch in enumerate(train_loader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(
            device_type=device.type, enabled=args.amp and device.type == "cuda"
        ):
            ret = model(batch, current_epoch=epoch)
            terms = {}
            for name, value in ret.items():
                if not name.endswith("_loss"):
                    continue
                warmup = (
                    args.completion_warmup_epochs
                    if name.startswith("completion")
                    else args.fusion_warmup_epochs
                    if name.startswith("fusion")
                    else args.modality_dropout_warmup_epochs
                    if name == "consistency_loss"
                    else 0
                )
                ramp = min(1.0, max(0.0, (epoch - 1) / warmup)) if warmup else 1.0
                terms[name] = value * ramp
            total = sum(terms.values())
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}, batch {iteration}"
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        for name, value in terms.items():
            stats[name] = stats.get(name, 0.0) + float(value.detach())
        steps += 1
        if (iteration + 1) % args.log_period == 0:
            logging.getLogger("ORBench").info(
                "epoch=%d batch=%d loss=%.5f",
                epoch,
                iteration + 1,
                float(total.detach()),
            )
    if not steps:
        raise ValueError("Empty training loader")
    means = {k: v / steps for k, v in stats.items()}
    if tensorboard_writer:
        for name, value in means.items():
            tensorboard_writer.add_scalar("Train/" + name, value, epoch)
    return means


def main():
    args = get_args()
    configure_runtime(args)
    seed_everything(args.seed)
    if args.output_dir == "logs":
        args.output_dir = str(
            Path("logs")
            / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.dataset_name)
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("ORBench", save_dir=args.output_dir, if_train=True)
    loaders = build_dataloader(args)
    train_loader, val_loaders, num_classes = loaders[0], loaders[1:-1], loaders[-1]
    args.architecture_version = ARCHITECTURE_VERSION
    args.checkpoint_selection = (
        "validation_R1" if val_loaders[0] is not None else "fixed_final_epoch"
    )
    save_train_configs(args.output_dir, args)
    save_run_manifest(args, Path(args.output_dir) / "run_manifest.json")
    model = build_model(args, num_classes).to(resolve_device(args.device))
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    checkpointer = Checkpointer(
        model, optimizer, scheduler, args.output_dir, logger=logger
    )
    start, best = 0, -float("inf")
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and next(model.parameters()).device.type == "cuda"
    )
    if args.resume:
        data = checkpointer.resume(args.resume_ckpt_file)
        start, best = data["epoch"], data.get("best_top1", best)
        if "scaler" in data:
            scaler.load_state_dict(data["scaler"])
        if "rng_torch" in data:
            torch.set_rng_state(data["rng_torch"])
        if "rng_python" in data:
            random.setstate(data["rng_python"])
        if "rng_numpy" in data:
            np.random.set_state(data["rng_numpy"])
        if torch.cuda.is_available() and data.get("rng_cuda"):
            torch.cuda.set_rng_state_all(data["rng_cuda"])
    logger.info(
        "Checkpoint selection: %s; no test-set model selection",
        args.checkpoint_selection,
    )
    with SummaryWriter(str(Path(args.output_dir) / "tensorboard")) as writer:
        for epoch in range(start + 1, args.num_epoch + 1):
            stats = train_one_epoch(
                args,
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                tensorboard_writer=writer,
                scaler=scaler,
            )
            scheduler.step()
            metadata = dict(
                epoch=epoch,
                best_top1=best,
                selection=args.checkpoint_selection,
                scaler=scaler.state_dict(),
                rng_torch=torch.get_rng_state(),
                rng_python=random.getstate(),
                rng_numpy=np.random.get_state(),
                rng_cuda=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            )
            improved = False
            if val_loaders[0] is not None and (
                epoch % args.eval_period == 0 or epoch == args.num_epoch
            ):
                val_args = copy.copy(args)
                val_args.eval_missing_rate = val_args.eval_gallery_missing_rate = 0.0
                val_args.protocol_dir = str(
                    Path(args.output_dir) / "validation" / f"epoch_{epoch}"
                )
                evaluator = evaluator_from_loaders(val_loaders, val_args)
                score = evaluator.eval(model)
                writer.add_scalar("Validation/R1", score, epoch)
                if score > best:
                    best = score
                    metadata["best_top1"] = best
                    improved = True
            metadata.update(
                rng_torch=torch.get_rng_state(),
                rng_python=random.getstate(),
                rng_numpy=np.random.get_state(),
                rng_cuda=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            )
            if improved:
                checkpointer.save("best", **metadata)
            checkpointer.save("last", **metadata)
            if epoch == args.num_epoch:
                checkpointer.save("final", **metadata)
                if val_loaders[0] is None:
                    checkpointer.save("best", **metadata)
            with open(
                Path(args.output_dir) / "training_metrics.jsonl", "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "effective_losses": stats,
                            "selection": args.checkpoint_selection,
                        }
                    )
                    + "\n"
                )
            logger.info("epoch=%d losses=%s", epoch, stats)


if __name__ == "__main__":
    main()
