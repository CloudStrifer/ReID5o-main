"""Strict versioned checkpoints; legacy architectures require explicit migration."""

from pathlib import Path
import logging
import torch
from model.build import ARCHITECTURE_VERSION


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=True,
        logger=None,
    ):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.save_dir, self.save_to_disk = save_dir, save_to_disk
        self.logger = logger or logging.getLogger("ORBench")

    def save(self, name, **metadata):
        if not self.save_dir or not self.save_to_disk:
            return
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "architecture_version": ARCHITECTURE_VERSION,
            "num_classes": self.model.num_classes,
            "model": self.model.state_dict(),
            "config": vars(self.model.args),
            **metadata,
        }
        if self.optimizer:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler:
            data["scheduler"] = self.scheduler.state_dict()
        path = Path(self.save_dir) / (name + ".pth")
        torch.save(data, path)
        self.logger.info("Saved %s", path)

    def load(self, f=None):
        if not f:
            raise ValueError("A checkpoint path is required")
        data = torch.load(f, map_location="cpu", weights_only=False)
        if data.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError(
                "Checkpoint is not MFCRAF v2. Original weights cannot be silently loaded into "
                "the new SPA-LoRA/completion/fusion architecture; retrain or explicitly migrate."
            )
        if data["num_classes"] != self.model.num_classes:
            raise ValueError(
                "Checkpoint classifier size disagrees with the dataset/config"
            )
        self.model.load_state_dict(data["model"], strict=True)
        return data

    def resume(self, f=None):
        data = self.load(f)
        if self.optimizer and "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        if self.scheduler and "scheduler" in data:
            self.scheduler.load_state_dict(data["scheduler"])
        return data


def load_for_inference(args, checkpoint, device):
    from model import build_model

    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if metadata.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Expected a retrained MFCRAF v2 checkpoint")
    for key in (
        "add_lora",
        "spa_experts",
        "spa_parts",
        "lora_r",
        "lora_layers",
        "use_missing_aware",
        "use_cross_modal_completion",
        "use_reliability_fusion",
    ):
        if getattr(args, key, None) != metadata["config"].get(key):
            raise ValueError(
                f"Inference config {key} differs from the training checkpoint"
            )
    model = build_model(args, metadata["num_classes"])
    model.load_state_dict(metadata["model"], strict=True)
    return model.to(device).eval()
