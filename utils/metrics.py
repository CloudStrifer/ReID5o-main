import json
import logging
from pathlib import Path
import torch
from torch.nn import functional as F
from .protocol import deterministic_masks, save_protocol
from datasets.person_data import QUERY_ORDER


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if similarity.ndim != 2 or not len(g_pids):
        raise ValueError("Empty gallery")
    indices = similarity.argsort(1, descending=True)
    matches = g_pids[indices].eq(q_pids[:, None])
    relevant = matches.sum(1)
    valid = relevant > 0
    if not valid.any():
        raise ValueError("No query has a relevant gallery identity")
    matches = matches[valid]
    cumulative = matches.cumsum(1)
    cmc = cumulative.clamp(max=1).float().mean(0)
    # Repeat the last rank for galleries smaller than the requested max_rank.
    if len(cmc) < max_rank:
        cmc = F.pad(cmc, (0, max_rank - len(cmc)), value=float(cmc[-1]))
    cmc = cmc[:max_rank] * 100
    if not get_mAP:
        return cmc, None, None, indices
    positions = torch.arange(1, similarity.shape[1] + 1, device=similarity.device)
    ap = ((cumulative / positions) * matches).sum(1) / relevant[valid]
    last = (matches * positions).max(1).values
    inp = relevant[valid] / last
    return cmc, ap.mean() * 100, inp.mean() * 100, indices


def score_features(
    qfeats, gfeats, qids, gids, get_mAP=True, chunk_size=128, device="cpu"
):
    """Bound ranking memory by chunk_size * gallery_size, rather than Q * G."""
    if chunk_size < 1 or not torch.isin(qids, gids).all():
        raise ValueError(
            "Ranking requires a positive chunk size and a match for every query"
        )
    gallery, gallery_ids = gfeats.to(device), gids.to(device)
    cmc_sum, ap_sum, inp_sum, count = torch.zeros(10, device=device), 0.0, 0.0, 0
    for start in range(0, len(qfeats), chunk_size):
        stop = min(start + chunk_size, len(qfeats))
        cmc, ap, inp, _ = rank(
            qfeats[start:stop].to(device) @ gallery.T,
            qids[start:stop].to(device),
            gallery_ids,
            get_mAP=get_mAP,
        )
        n = stop - start
        cmc_sum += cmc * n
        if get_mAP:
            ap_sum += float(ap) * n
            inp_sum += float(inp) * n
        count += n
    if not count:
        raise ValueError("Empty query set")
    return (
        cmc_sum / count,
        ap_sum / count if get_mAP else None,
        inp_sum / count if get_mAP else None,
    )


class Evaluator:
    def __init__(
        self,
        gallery_loader,
        get_mAP=True,
        args=None,
        query_loaders=None,
        **legacy_loaders,
    ):
        self.gallery_loader, self.get_mAP, self.args = gallery_loader, get_mAP, args
        self.query_loaders = query_loaders or {
            task: legacy_loaders.get(task.lower().replace("+", "_") + "_query_loader")
            for task in QUERY_ORDER
        }
        self.results = {}

    @torch.no_grad()
    def extract(self, model, loader, task, rate=0.0):
        if loader is None:
            raise ValueError(f"No loader for {task}")
        device = next(model.parameters()).device
        all_ids, all_feats, records = [], [], []
        weight_sum = torch.zeros(5)
        count = 0
        seed = getattr(self.args, "eval_seed", 42)
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            available = batch["modality_available"]
            observed = deterministic_masks(
                available, batch["sample_index"], rate, seed, task
            )
            feats, info = model.encode_multimodal(
                batch, observed, available, return_details=True
            )
            all_ids.append(batch["pids"].cpu())
            all_feats.append(feats.cpu())
            weight_sum += info["fusion_weights"].float().sum(0).cpu()
            count += len(feats)
            for i, index in enumerate(batch["sample_index"].tolist()):
                records.append(
                    {
                        "index": index,
                        "pid": int(batch["pids"][i]),
                        "available": available[i].tolist(),
                        "observed": observed[i].tolist(),
                    }
                )
        output_dir = getattr(self.args, "protocol_dir", "") or str(
            Path(getattr(self.args, "output_dir", "logs")) / "protocols"
        )
        filename = task.replace("+", "_") + f"_rate_{rate:g}_seed_{seed}.json"
        save_protocol(Path(output_dir) / filename, records, rate, seed, task)
        # Report all five modalities. Nonexistent modalities retain exactly zero weight.
        Path(output_dir, filename.replace(".json", "_weights.json")).write_text(
            json.dumps(
                {
                    "modalities": ["RGB", "NIR", "CP", "SK", "TEXT"],
                    "mean_weights": (weight_sum / max(count, 1)).tolist(),
                    "sample_count": count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not all_ids:
            raise ValueError(f"Empty {task} set")
        return torch.cat(all_ids), torch.cat(all_feats)

    def eval(self, model):
        model.eval()
        gids, gfeats = self.extract(
            model,
            self.gallery_loader,
            "RGB_GALLERY",
            getattr(self.args, "eval_gallery_missing_rate", 0.0),
        )
        for task, loader in self.query_loaders.items():
            if loader is None:
                continue
            qids, qfeats = self.extract(
                model, loader, task, getattr(self.args, "eval_missing_rate", 0.0)
            )
            if not torch.isin(qids, gids).all():
                raise ValueError(f"{task} has identities without gallery matches")
            cmc, ap, inp = score_features(
                qfeats,
                gfeats,
                qids,
                gids,
                self.get_mAP,
                getattr(self.args, "metric_chunk_size", 128),
                next(model.parameters()).device,
            )
            self.results[task] = {
                "R1": float(cmc[0]),
                "R5": float(cmc[4]),
                "R10": float(cmc[9]),
                "mAP": float(ap) if ap is not None else None,
                "mINP": float(inp) if inp is not None else None,
            }
            logging.getLogger("ORBench.eval").info("%s: %s", task, self.results[task])
        if not self.results:
            raise ValueError("No query sets supplied")
        return sum(x["R1"] for x in self.results.values()) / len(self.results)


def evaluator_from_loaders(loaders, args):
    return Evaluator(
        loaders[0],
        get_mAP=args.compute_mAP,
        args=args,
        query_loaders=dict(zip(QUERY_ORDER, loaders[1:])),
    )
