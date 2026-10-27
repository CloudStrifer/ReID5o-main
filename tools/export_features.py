"""Export actual embeddings and paired real/completed targets without fabricating plots."""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.options import get_args
from utils.iotools import load_train_configs
from utils.checkpoint import load_for_inference
from utils.protocol import deterministic_masks
from datasets import build_dataloader
from datasets.person_data import QUERY_ORDER, MODALITIES
from train import resolve_device, configure_runtime


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--task", default="TEXT+SK", choices=QUERY_ORDER)
    p.add_argument("--mask_modalities", nargs="*", default=[], choices=MODALITIES)
    p.add_argument("--missing_rate", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    cli = p.parse_args()
    args = get_args([])
    vars(args).update(dict(load_train_configs(cli.config_file)))
    args.training = False
    configure_runtime(args)
    model = load_for_inference(
        args,
        cli.checkpoint or str(Path(args.output_dir) / "best.pth"),
        resolve_device(cli.device),
    )
    loader = build_dataloader(args)[1 + QUERY_ORDER.index(cli.task)]
    if loader is None:
        raise ValueError("The requested query combination is absent")
    buffers = {"features": [], "pids": [], "observed": [], "weights": []}
    paired = {m: {"real": [], "completed": [], "pids": []} for m in MODALITIES}
    count = 0
    for batch in loader:
        b = min(len(batch["pids"]), cli.max_samples - count)
        if b <= 0:
            break
        batch = {k: v[:b].to(next(model.parameters()).device) for k, v in batch.items()}
        available = batch["modality_available"]
        mask = deterministic_masks(
            available, batch["sample_index"], cli.missing_rate, cli.seed, cli.task
        )
        for m in cli.mask_modalities:
            mask[:, MODALITIES.index(m)] = False
        if not mask.any(1).all():
            raise ValueError("Explicit masking removed all query observations")
        output, details = model.encode_multimodal(batch, mask, available, True)
        _, teacher = model.encode_multimodal(batch, available, available, True)
        for key, value in [
            ("features", output),
            ("pids", batch["pids"]),
            ("observed", mask),
            ("weights", details["fusion_weights"]),
        ]:
            buffers[key].append(value.cpu().numpy())
        for k, m in enumerate(MODALITIES):
            valid = available[:, k] & ~mask[:, k]
            if m in details["predictions"] and valid.any():
                paired[m]["real"].append(
                    teacher["pooled_features"][m][valid].cpu().numpy()
                )
                paired[m]["completed"].append(
                    details["predictions"][m][valid].cpu().numpy()
                )
                paired[m]["pids"].append(batch["pids"][valid].cpu().numpy())
        count += b
    result = {k: np.concatenate(v) for k, v in buffers.items() if v}
    for m, groups in paired.items():
        for k, v in groups.items():
            if v:
                result[m + "_" + k] = np.concatenate(v)
    result["modalities"] = np.array(MODALITIES)
    result["query_task"] = np.array(cli.task)
    result["seed"] = np.array(cli.seed)
    Path(cli.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cli.output, **result)
    print("Exported", count, "samples to", cli.output)


if __name__ == "__main__":
    main()
