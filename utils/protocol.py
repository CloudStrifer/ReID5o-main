"""Versioned, batch/order-independent missing-modality evaluation masks."""

import hashlib
import json
import random
from pathlib import Path
import torch

PROTOCOL_VERSION = "bernoulli-repair-v1"


def deterministic_masks(available, sample_indices, rate, seed, namespace="query"):
    if not 0 <= rate <= 1:
        raise ValueError("Missing rate must lie in [0,1]")
    available = available.bool()
    masks = []
    for row, index in zip(
        available.cpu().tolist(), torch.as_tensor(sample_indices).cpu().tolist()
    ):
        candidates = [i for i, x in enumerate(row) if x]
        if not candidates:
            raise ValueError("Cannot degrade an empty sample")
        digest = hashlib.sha256(
            f"{PROTOCOL_VERSION}:{seed}:{namespace}:{index}".encode()
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        mask = [value and rng.random() >= rate for value in row]
        if not any(mask):
            mask[rng.choice(candidates)] = True
        masks.append(mask)
    return torch.tensor(masks, device=available.device, dtype=torch.bool)


def save_protocol(path, records, rate, seed, namespace):
    available = sum(sum(x["available"]) for x in records)
    retained = sum(sum(x["observed"]) for x in records)
    result = {
        "protocol": PROTOCOL_VERSION,
        "seed": seed,
        "namespace": namespace,
        "requested_rate": rate,
        "actual_deleted_fraction": 1 - retained / available if available else 0.0,
        "repair": "uniformly retain one originally available modality if all were dropped",
        "samples": records,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
