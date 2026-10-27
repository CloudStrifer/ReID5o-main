"""Small, explicit provenance records for new experiments."""

import hashlib
import json
import platform
import sys
from pathlib import Path
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_run_manifest(args, destination):
    root = Path(__file__).resolve().parents[1]
    sources = [root / x for x in ("train.py", "test.py", "infer.py")]
    for directory in ("model", "datasets", "utils", "solver"):
        sources.extend((root / directory).glob("*.py"))
    data = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "argv": sys.argv,
        "config": vars(args),
        "source_sha256": {
            p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(sources)
        },
    }
    split = getattr(args, "split_file", None)
    if split:
        data["split_manifest_sha256"] = sha256_file(split)
    Path(destination).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
