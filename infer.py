"""Interactive retrieval through exactly the same MFCRAF encoder as test.py."""

import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image
from datasets import build_dataloader
from datasets.build import build_transforms
from datasets.person_data import tokenize
from utils.simple_tokenizer import SimpleTokenizer
from utils.iotools import load_train_configs, read_image
from utils.options import get_args
from utils.checkpoint import load_for_inference
from train import resolve_device, configure_runtime


class ImageRetriever:
    def __init__(self, config_path, device="auto", checkpoint=None):
        self.args = get_args([])
        vars(self.args).update(dict(load_train_configs(config_path)))
        self.args.training = False
        configure_runtime(self.args)
        self.device = resolve_device(device)
        checkpoint = checkpoint or str(Path(self.args.output_dir) / "best.pth")
        self.model = load_for_inference(self.args, checkpoint, self.device)
        self.transform = build_transforms(self.args.img_size, is_train=False)
        self.tokenizer = SimpleTokenizer()

    @torch.no_grad()
    def extract_gallery_features(self, gallery_loader):
        ids, features, paths = [], [], []
        for batch in gallery_loader:
            indices = batch["sample_index"].tolist()
            inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            features.append(self.model.encode_multimodal(inputs).cpu())
            ids.append(batch["pids"])
            dataset = gallery_loader.dataset
            if hasattr(dataset, "img_paths"):
                paths.extend(dataset.img_paths[i] for i in indices)
            else:
                paths.extend(
                    next(
                        (
                            v
                            for m, v in zip(dataset.modalities, dataset.samples[i][1:])
                            if m != "TEXT" and v
                        ),
                        None,
                    )
                    for i in indices
                )
        return torch.cat(ids), torch.cat(features), paths

    @torch.no_grad()
    def extract_query_features(
        self, nir_path=None, cp_path=None, sk_path=None, text=None, rgb_path=None
    ):
        inputs = {}
        for key, path in [
            ("rgbs", rgb_path),
            ("nirs", nir_path),
            ("cps", cp_path),
            ("sks", sk_path),
        ]:
            if path:
                inputs[key] = (
                    self.transform(read_image(path)).unsqueeze(0).to(self.device)
                )
        if text and text.strip():
            inputs["caption_ids"] = (
                tokenize(text, self.tokenizer, self.args.text_length)
                .unsqueeze(0)
                .to(self.device)
            )
        if not inputs:
            raise ValueError("At least one query modality is required")
        return self.model.encode_multimodal(inputs).cpu()

    def retrieve(self, query_features, gallery_features, topk=10):
        return (
            (query_features @ gallery_features.T)
            .topk(min(topk, len(gallery_features)), dim=1)
            .indices
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--gallery_file", default=None)
    for name in ("rgb_path", "nir_path", "cp_path", "sk_path", "text"):
        p.add_argument("--" + name, default=None)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--output", default="retrieval.png")
    cli = p.parse_args()
    retriever = ImageRetriever(cli.config_file, cli.device, cli.checkpoint)
    if cli.gallery_file:
        retriever.args.gallery_file = cli.gallery_file
    loaders = build_dataloader(retriever.args)
    ids, features, paths = retriever.extract_gallery_features(loaders[0])
    query = retriever.extract_query_features(
        cli.nir_path, cli.cp_path, cli.sk_path, cli.text, cli.rgb_path
    )
    indices = retriever.retrieve(query, features, cli.topk)[0].tolist()
    fig, axes = plt.subplots(
        1, len(indices), figsize=(2 * len(indices), 4), squeeze=False
    )
    for rank, (ax, index) in enumerate(zip(axes[0], indices), 1):
        if paths[index]:
            ax.imshow(Image.open(paths[index]))
        else:
            ax.text(0.5, 0.5, "Text gallery", ha="center", va="center")
        ax.set_title(f"Rank {rank}, ID {int(ids[index])}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(cli.output, dpi=200)
    plt.close(fig)
    print("Saved", str(Path(cli.output).resolve()))


if __name__ == "__main__":
    main()
