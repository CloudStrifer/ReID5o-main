"""Explicit splits and availability. RGB is never a substitute for a missing modality."""

import itertools
import json
import logging
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.simple_tokenizer import SimpleTokenizer
from utils.iotools import read_image

MODALITIES = ("RGB", "NIR", "CP", "SK", "TEXT")
INPUT_KEYS = ("rgbs", "nirs", "cps", "sks", "caption_ids")
QUERY_ORDER = list(MODALITIES[1:])
for count in (2, 3, 4):
    for combo in itertools.combinations(MODALITIES[1:], count):
        for first in combo:
            QUERY_ORDER.append(
                "+".join((first,) + tuple(x for x in combo if x != first))
            )


def tokenize(caption, tokenizer, text_length=77):
    ids = [tokenizer.encoder["<|startoftext|>"]] + tokenizer.encode(caption)
    ids = ids[: text_length - 1] + [tokenizer.encoder["<|endoftext|>"]]
    result = torch.zeros(text_length, dtype=torch.long)
    result[: len(ids)] = torch.tensor(ids)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class PersonDataset:
    def __init__(self, root, name, split_file=None, missing_sketch_policy="mask"):
        self.name, self.root = name, Path(root)
        self.missing_sketch_policy, self.missing_sketches = missing_sketch_policy, []
        self.train_caption_choices = {}
        self.train_modality_pools = {}
        if name == "ORBench":
            self._load_orbench()
        else:
            self._load_three_modal(split_file)
        self.train_id_container = {x[0] for x in self.train}
        self.test_id_container = set(self.test["gallery_pids"])
        logging.getLogger("ORBench.dataset").info(
            "%s: train IDs=%d samples=%d, test IDs=%d, missing sketches=%d",
            name,
            len(self.train_id_container),
            len(self.train),
            len(self.test_id_container),
            len(self.missing_sketches),
        )

    @staticmethod
    def _check_disjoint(parts):
        for a, b in itertools.combinations(parts, 2):
            if parts[a] & parts[b]:
                raise ValueError(f"Identity overlap between {a} and {b}")

    def _load_three_modal(self, split_file):
        folder, annotation, skfolder, key = {
            "CUHK-PEDES": ("CUHK-PEDES", "caption_all.json", "CUHK", "file_path"),
            "ICFG-PEDES": ("ICFG-PEDES", "ICFG-PEDES.json", "ICFG", "file_path"),
            "RSTPReid": ("RSTPReid", "data_captions.json", "RSTPReid", "img_path"),
        }[self.name]
        candidates = [self.root / folder, self.root / folder / folder, self.root]
        candidates += [p / folder for p in self.root.iterdir() if p.is_dir()]
        valid = list(dict.fromkeys(p for p in candidates if (p / annotation).is_file()))
        if len(valid) != 1:
            raise FileNotFoundError(
                f"Expected one {annotation} under {self.root}; found {valid}"
            )
        base = valid[0]
        rows = read_json(base / annotation)
        self.annotation_paths = [str(base / annotation)]
        manifest = Path(split_file) if split_file else base / "reid_raw.json"
        if split_file and not manifest.exists():
            raise FileNotFoundError(manifest)
        if manifest.exists():
            self.annotation_paths.append(str(manifest))
            source = read_json(manifest)
            meta = (
                {
                    x.get("file_path", x.get("img_path", "")).replace(chr(92), "/"): x
                    for x in source
                }
                if isinstance(source, list)
                else {
                    k.replace(chr(92), "/"): {"split": v} if isinstance(v, str) else v
                    for k, v in source.items()
                }
            )
            for row in rows:
                path = row[key].replace(chr(92), "/")
                if path not in meta:
                    raise ValueError(f"Split manifest has no entry for {path}")
                row.update(
                    {k: v for k, v in meta[path].items() if k in ("split", "id")}
                )
        if any("split" not in x for x in rows):
            raise ValueError(
                "CUHK-PEDES needs an official split manifest: --split_file or "
                "CUHK-PEDES/reid_raw.json. Source directory names are not benchmark splits."
            )
        parts = {
            s: [x for x in rows if x["split"] == s] for s in ("train", "val", "test")
        }
        if sum(map(len, parts.values())) != len(rows):
            raise ValueError("Unknown split name; expected train, val or test")
        self._check_disjoint({s: {x["id"] for x in v} for s, v in parts.items()})
        skroot = self.root / "sketch" / "aliyun" / skfolder / "imgs"
        if not skroot.exists():
            nested = list(self.root.glob(f"*/sketch/aliyun/{skfolder}/imgs"))
            if len(nested) != 1:
                raise FileNotFoundError(
                    f"Sketch directory missing or ambiguous: {skroot}"
                )
            skroot = nested[0]
        sketches = {}
        for path in skroot.rglob("*"):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            stem = path.relative_to(skroot).with_suffix("").as_posix().casefold()
            if stem in sketches:
                raise ValueError(f"Ambiguous sketch stem: {stem}")
            sketches[stem] = str(path)

        def paths(row):
            relative = Path(row[key].replace(chr(92), "/"))
            sk = sketches.get(relative.with_suffix("").as_posix().casefold())
            if sk is None:
                self.missing_sketches.append(relative.as_posix())
                if self.missing_sketch_policy == "error":
                    raise FileNotFoundError(f"No sketch paired with {relative}")
            return str(base / "imgs" / relative), sk

        labels = {
            pid: i for i, pid in enumerate(sorted({x["id"] for x in parts["train"]}))
        }
        self.train = []
        for i, row in enumerate(parts["train"]):
            rgb, sk = paths(row)
            captions = row.get("captions", [])
            captions = captions if isinstance(captions, list) else [captions]
            captions = [caption for caption in captions if caption]
            self.train_caption_choices[rgb] = captions
            cap = captions[0] if captions else ""
            self.train.append((labels[row["id"]], i, rgb, None, None, sk, cap or ""))

        def evaluation(rows):
            gallery, pids, queries = [], [], {k: [] for k in QUERY_ORDER}
            for row in rows:
                rgb, sk = paths(row)
                pid = row["id"]
                gallery.append(rgb)
                pids.append(pid)
                caps = row.get("captions", [])
                caps = caps if isinstance(caps, list) else [caps]
                if sk:
                    queries["SK"].append((pid, sk))
                for cap in caps:
                    if cap:
                        queries["TEXT"].append((pid, cap))
                        queries["TEXT+SK"].append((pid, cap, sk))
                        queries["SK+TEXT"].append((pid, sk, cap))
            return {"gallery_paths": gallery, "gallery_pids": pids, "queries": queries}

        self.val = evaluation(parts["val"]) if parts["val"] else None
        self.test = evaluation(parts["test"])

    def _load_orbench(self):
        base = self.root / "ORBench"
        self.annotation_paths = [
            str(base / "train_annos.json"),
            str(base / "test_gallery_and_queries.json"),
        ]
        rows = read_json(base / "train_annos.json")
        labels = {pid: i for i, pid in enumerate(sorted({int(x["id"]) for x in rows}))}
        pools = {m: {} for m in ("nir", "cp", "sk")}
        for m in pools:
            for folder in sorted((base / m).iterdir()):
                if folder.is_dir():
                    pools[m][folder.name] = [
                        str(x)
                        for x in sorted(folder.iterdir())
                        if x.suffix.lower() in (".jpg", ".png", ".bmp", ".jpeg")
                    ]
        self.train_modality_pools = pools
        self.train = []
        for i, row in enumerate(rows):
            identity = row["file_path"].replace(chr(92), "/").split("/")[1]
            aux = [random.choice(pools[m][identity]) for m in ("nir", "cp", "sk")]
            self.train.append(
                (
                    labels[int(row["id"])],
                    i,
                    str(base / row["file_path"]),
                    *aux,
                    row["caption"],
                )
            )
        raw = read_json(base / "test_gallery_and_queries.json")
        gallery = raw.pop("RGB_GALLERY")
        self._check_disjoint(
            {
                "train": {int(x["id"]) for x in rows},
                "test": {int(x[0]) for x in gallery},
            }
        )
        queries = {}
        for task, samples in raw.items():
            mods = task.split("+")
            queries[task] = [
                (
                    int(x[0]),
                    *(v if m == "TEXT" else str(base / v) for m, v in zip(mods, x[1:])),
                )
                for x in samples
            ]
        self.test = {
            "gallery_paths": [str(base / x[1]) for x in gallery],
            "gallery_pids": [int(x[0]) for x in gallery],
            "queries": queries,
        }
        self.val = None


class TrainDataset(Dataset):
    def __init__(
        self,
        dataset,
        transform,
        text_length=77,
        caption_choices=None,
        modality_pools=None,
    ):
        self.dataset = list(dataset)
        self.transform, self.text_length = transform, text_length
        self.caption_choices = caption_choices or {}
        self.modality_pools = modality_pools or {}
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def random_sampling(self):
        for index, record in enumerate(self.dataset):
            record = list(record)
            choices = self.caption_choices.get(record[2], [])
            if choices:
                record[6] = random.choice(choices)
            if self.modality_pools:
                identity = record[2].replace(chr(92), "/").split("/")[-2]
                for column, modality in enumerate(("nir", "cp", "sk"), 3):
                    choices = self.modality_pools[modality][identity]
                    if not choices:
                        raise ValueError(
                            f"No {modality} sample for identity {identity}"
                        )
                    record[column] = random.choice(choices)
            self.dataset[index] = tuple(record)

    def __getitem__(self, index):
        pid, image_id, *values = self.dataset[index]
        result = {
            "pids": pid,
            "image_ids": image_id,
            "modality_available": torch.tensor([bool(x) for x in values]),
        }
        blank = torch.zeros_like(self.transform(Image.new("RGB", (128, 384))))
        for k, v in zip(INPUT_KEYS[:4], values[:4]):
            result[k] = self.transform(read_image(v)) if v else blank.clone()
        result["caption_ids"] = tokenize(values[4], self.tokenizer, self.text_length)
        return result


class QueryDataset(Dataset):
    def __init__(self, samples, modalities, transform, text_length=77):
        self.samples, self.modalities, self.transform = samples, modalities, transform
        self.text_length, self.tokenizer = text_length, SimpleTokenizer()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pid, *values = self.samples[index]
        result = {"pids": pid, "sample_index": index}
        available = torch.zeros(5, dtype=torch.bool)
        for m, v in zip(self.modalities, values):
            i = MODALITIES.index(m)
            available[i] = bool(v)
            if i == 4:
                result[INPUT_KEYS[i]] = tokenize(
                    v or "", self.tokenizer, self.text_length
                )
            else:
                result[INPUT_KEYS[i]] = (
                    self.transform(read_image(v))
                    if v
                    else torch.zeros_like(self.transform(Image.new("RGB", (128, 384))))
                )
        if not available.any():
            raise ValueError(f"Query {index} has no observed modality")
        result["modality_available"] = available
        return result


class GalleryDataset(QueryDataset):
    def __init__(self, image_pids, img_paths, transform, text_length=77):
        self.img_paths = img_paths
        super().__init__(
            list(zip(image_pids, img_paths)), ["RGB"], transform, text_length
        )
