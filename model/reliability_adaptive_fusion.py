"""Reliability-adaptive fusion, matching paper equations (13)--(20)."""

import torch
from torch import nn
from torch.nn import functional as F
from .cross_modal_completion import MODALITIES


class ReliabilityAdaptiveFusion(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_modalities=5,
        hidden_dim=256,
        temperature=1.0,
        entropy_target=0.3,
        margin=0.5,
        use_quality_indicators=True,
        **unused,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Fusion temperature must be positive")
        self.temperature, self.entropy_target, self.margin = (
            temperature,
            entropy_target,
            margin,
        )
        self.use_quality_indicators = use_quality_indicators
        self.confidence = nn.Linear(embed_dim, 1)
        self.reliability_networks = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(
                        embed_dim + (3 if use_quality_indicators else 0) + 1, hidden_dim
                    ),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for m in MODALITIES
            }
        )
        self.modality_bias = nn.Parameter(torch.zeros(5))
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, features, valid, generated, return_weights=True):
        reference = next(iter(features.values()))
        stacked = torch.stack(
            [features.get(m, torch.zeros_like(reference)) for m in MODALITIES], 1
        )
        scores = []
        for k, m in enumerate(MODALITIES):
            x = stacked[:, k]
            parts = [x]
            if self.use_quality_indicators:
                normalized = F.normalize(x, dim=-1)
                parts += [
                    normalized.var(-1, unbiased=False, keepdim=True),
                    x.norm(dim=-1, keepdim=True),
                    self.confidence(x).sigmoid(),
                ]
            parts += [generated[:, k, None].to(x.dtype)]
            scores.append(
                self.reliability_networks[m](torch.cat(parts, -1)).squeeze(-1)
                + self.modality_bias[k]
            )
        raw_scores = torch.stack(scores, 1)
        if not valid.any(1).all():
            raise ValueError("Fusion has an empty sample")
        weights = (
            (raw_scores / self.temperature).masked_fill(~valid, -torch.inf).softmax(-1)
        )
        weighted = (weights[:, :, None] * stacked).sum(1)
        fused = self.output_proj(weighted)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1).mean()
        sparsity = F.relu(entropy - self.entropy_target)
        uncertainty = (
            (weights * (generated.to(weights.dtype) + F.relu(self.margin - raw_scores)))
            .sum(1)
            .mean()
        )
        return fused, {
            "fusion_weights": weights,
            "reliability_scores": raw_scores,
            "weighted_features": weighted,
            "sparsity_loss": sparsity,
            "uncertainty_loss": uncertainty,
        }
