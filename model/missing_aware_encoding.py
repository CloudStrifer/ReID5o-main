"""Per-sample modality masks and semantic state embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class ModalityDropout(nn.Module):
    def __init__(
        self, num_modalities=5, min_keep=1, dropout_prob=0.3, keep_rgb_prob=0.9
    ):
        super().__init__()
        if not 0 <= dropout_prob <= 1 or not 0 <= keep_rgb_prob <= 1:
            raise ValueError("Invalid dropout probability")
        if min_keep < 1:
            raise ValueError("min_keep must be positive")
        self.num_modalities, self.min_keep = num_modalities, min_keep
        self.dropout_prob, self.keep_rgb_prob = dropout_prob, keep_rgb_prob

    def forward(self, available, training=True, generator=None):
        if available.ndim != 2 or available.shape[1] != self.num_modalities:
            raise ValueError("Availability must be [batch,modalities]")
        available = available.bool()
        if not available.any(1).all():
            raise ValueError("A sample has no available modality")
        if not training:
            return available.clone()
        probs = torch.full(
            available.shape, 1 - self.dropout_prob, device=available.device
        )
        probs[:, 0] = self.keep_rgb_prob
        keep = (
            torch.rand(probs.shape, device=available.device, generator=generator)
            < probs
        ) & available
        for i in range(len(keep)):
            target = min(self.min_keep, int(available[i].sum()))
            needed = target - int(keep[i].sum())
            if needed > 0:
                candidates = torch.where(available[i] & ~keep[i])[0]
                order = torch.randperm(
                    len(candidates), device=keep.device, generator=generator
                )
                keep[i, candidates[order[:needed]]] = True
        return keep


class MissingAwareEncoder(nn.Module):
    def __init__(
        self, embed_dim, num_modalities=5, vision_num_tokens=193, text_num_tokens=77
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(num_modalities, embed_dim)
        self.mask_embedding = nn.Embedding(2, embed_dim)
        self.missing_tokens = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        text_num_tokens if i == 4 else vision_num_tokens, embed_dim
                    )
                    * 0.02
                )
                for i in range(num_modalities)
            ]
        )

    def forward(self, tokens, modality_idx, observed):
        b = observed.shape[0]
        missing = self.missing_tokens[modality_idx].unsqueeze(0).expand(b, -1, -1)
        if tokens is None:
            tokens = torch.zeros_like(missing)
        if tokens.shape != missing.shape:
            raise ValueError("Token shape disagrees with configured resolution")
        body = torch.where(observed[:, None, None], tokens, missing)
        return (
            body
            + self.type_embedding.weight[modality_idx]
            + self.mask_embedding(observed.long())[:, None, :]
        )


class ConsistencyConstraint(nn.Module):
    def forward(self, full_modal_embeds, subset_embeds):
        return (
            1 - F.cosine_similarity(full_modal_embeds.detach(), subset_embeds, dim=-1)
        ).mean()
