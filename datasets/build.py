import torch
from pathlib import Path
from .person_data import read_json
from utils.provenance import sha256_file
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .person_data import (
    PersonDataset,
    TrainDataset,
    QueryDataset,
    GalleryDataset,
    QUERY_ORDER,
)


def canonical_name(name):
    name = name.replace("-3Modal", "")
    return {
        "CUHK_PEDES": "CUHK-PEDES",
        "ICFG_PEDES": "ICFG-PEDES",
        "RSTP-Reid": "RSTPReid",
    }.get(name, name)


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    steps = [T.Resize(tuple(img_size))]
    if is_train:
        steps.append(T.RandomHorizontalFlip())
    if aug and is_train:
        steps += [T.Pad(10), T.RandomCrop(tuple(img_size))]
    steps += [
        T.ToTensor(),
        T.Normalize(
            [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        ),
    ]
    if aug and is_train:
        steps.append(T.RandomErasing())
    return T.Compose(steps)


def collate(batch):
    return torch.utils.data.default_collate(batch)


def build_dataloader(args, tranforms=None):
    ds = PersonDataset(
        args.root_dir,
        canonical_name(args.dataset_name),
        getattr(args, "split_file", None),
        getattr(args, "missing_sketch_policy", "mask"),
    )
    args.num_classes = len(ds.train_id_container)
    args.dataset_manifest = {
        "annotations": [
            {"path": p, "sha256": sha256_file(p)} for p in ds.annotation_paths
        ],
        "train_identities": args.num_classes,
        "test_identities": len(ds.test_id_container),
        "missing_sketches": len(ds.missing_sketches),
    }
    transform = tranforms or build_transforms(args.img_size, is_train=False)
    training = args.training
    if training:
        if getattr(args, "val_dataset", "val") == "test":
            raise ValueError(
                "Test data cannot select checkpoints. Use --val_dataset val or none."
            )
        evaluation = ds.val if getattr(args, "val_dataset", "val") == "val" else None
    else:
        evaluation = ds.test
    loaders = [None] * (1 + len(QUERY_ORDER))
    if evaluation and evaluation["gallery_pids"]:
        gallery_set = GalleryDataset(
            evaluation["gallery_pids"],
            evaluation["gallery_paths"],
            transform,
            args.text_length,
        )
        if getattr(args, "gallery_file", None) and not training:
            manifest_path = Path(args.gallery_file)
            manifest = read_json(manifest_path)
            modalities = manifest["modalities"]
            if not modalities or len(set(modalities)) != len(modalities):
                raise ValueError("Gallery modalities must be unique and nonempty")
            from .person_data import MODALITIES

            if set(modalities) - set(MODALITIES):
                raise ValueError("Unknown gallery modality")
            samples = []
            for pid, *values in manifest["samples"]:
                if len(values) != len(modalities):
                    raise ValueError("Gallery sample has wrong modality count")
                if pid not in ds.test_id_container:
                    raise ValueError(
                        "Custom gallery identity is outside the test split"
                    )
                values = [
                    str(manifest_path.parent / value)
                    if value and m != "TEXT"
                    else value
                    for m, value in zip(modalities, values)
                ]
                samples.append((pid, *values))
            gallery_set = QueryDataset(samples, modalities, transform, args.text_length)
        loaders[0] = DataLoader(
            gallery_set, args.test_batch_size, num_workers=args.num_workers
        )
        for i, task in enumerate(QUERY_ORDER, 1):
            rows = evaluation["queries"].get(task, [])
            if rows:
                loaders[i] = DataLoader(
                    QueryDataset(rows, task.split("+"), transform, args.text_length),
                    args.test_batch_size,
                    num_workers=args.num_workers,
                )
    if training:
        train = TrainDataset(
            ds.train,
            build_transforms(args.img_size, args.img_aug),
            args.text_length,
            caption_choices=ds.train_caption_choices,
            modality_pools=ds.train_modality_pools,
        )
        return (
            DataLoader(
                train,
                args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate,
            ),
            *loaders,
            args.num_classes,
        )
    return (*loaders, args.num_classes)
