"""Cross-attention feature completion with real-target and cycle supervision."""

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("RGB", "NIR", "CP", "SK", "TEXT")


class ModalityFeatureGenerator(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    embed_dim,
                    num_heads,
                    embed_dim * 4,
                    dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, context, context_mask, query_prior=None):
        query = self.query_token.expand(len(context), -1, -1)
        if query_prior is not None:
            query = query + query_prior[:, None, :]
        for layer in self.layers:
            query = layer(query, context, memory_key_padding_mask=~context_mask)
        return self.output_proj(query[:, 0])


class CrossModalCompletionModule(nn.Module):
    def __init__(
        self, embed_dim, num_modalities=5, num_heads=8, num_layers=2, dropout=0.1
    ):
        super().__init__()
        self.modality_names = list(MODALITIES)
        self.modality_context_proj = nn.ModuleDict(
            {m: nn.Linear(embed_dim, embed_dim) for m in MODALITIES}
        )
        self.generators = nn.ModuleDict(
            {
                m: ModalityFeatureGenerator(embed_dim, num_heads, num_layers, dropout)
                for m in MODALITIES
            }
        )

    def generate(self, features, observed, active, query_priors=None):
        if not observed.any(1).all():
            raise ValueError("Completion needs an observed context for every sample")
        context = torch.stack(
            [self.modality_context_proj[m](features[m]) for m in MODALITIES], 1
        )
        predictions = {}
        for k, m in enumerate(MODALITIES):
            if active[k]:
                prior = query_priors[m] if query_priors is not None else None
                predictions[m] = self.generators[m](context, observed, prior)
        return predictions

    def complete(self, features, observed, active, query_priors=None):
        predictions = self.generate(features, observed, active, query_priors)
        completed = {
            m: torch.where(observed[:, k, None], features[m], predictions[m])
            for k, m in enumerate(MODALITIES)
            if active[k]
        }
        return completed, predictions

    def reconstruction_loss(self, predictions, targets, masked):
        # Eq. (10): first average targets within a sample, then the full batch.
        # An empty target set contributes zero rather than changing the batch denominator.
        per_sample = next(iter(targets.values())).new_zeros(len(masked))
        for k, modality in enumerate(MODALITIES):
            rows = torch.where(masked[:, k])[0]
            if not len(rows):
                continue
            if modality not in predictions:
                raise ValueError(f"No prediction for masked target {modality}")
            distance = 1 - F.cosine_similarity(
                predictions[modality][rows], targets[modality][rows].detach(), dim=-1
            )
            per_sample = per_sample.index_add(0, rows, distance)
        counts = masked.sum(1).clamp_min(1)
        return (per_sample / counts).mean()

    def cycle_loss(self, features, available, active, query_priors=None, max_pairs=3):
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        # The paper defines a separate sampled pair set P_i for every sample.
        selected = torch.zeros(
            len(available), 5, 5, dtype=torch.bool, device=available.device
        )
        for i, row in enumerate((available & active).cpu().tolist()):
            candidates = [
                (a, b)
                for a in range(5)
                for b in range(5)
                if a != b and row[a] and row[b]
            ]
            for index in torch.randperm(len(candidates))[:max_pairs].tolist():
                a, b = candidates[index]
                selected[i, a, b] = True
        per_sample = next(iter(features.values())).new_zeros(len(available))
        for a, source in enumerate(MODALITIES):
            for b, target in enumerate(MODALITIES):
                if a == b or not active[a] or not active[b]:
                    continue
                rows = selected[:, a, b]
                if not rows.any():
                    continue
                source_features = {m: x[rows] for m, x in features.items()}
                mask = torch.zeros(
                    int(rows.sum()), 5, dtype=torch.bool, device=available.device
                )
                mask[:, a] = True
                context = torch.stack(
                    [
                        self.modality_context_proj[m](source_features[m])
                        for m in MODALITIES
                    ],
                    1,
                )
                intermediate = self.generators[target](
                    context,
                    mask,
                    query_priors[target][rows] if query_priors is not None else None,
                )
                # Only the generated intermediate is observed on the reverse path.
                reverse = {m: torch.zeros_like(intermediate) for m in MODALITIES}
                reverse[target] = intermediate
                mask[:] = False
                mask[:, b] = True
                context = torch.stack(
                    [self.modality_context_proj[m](reverse[m]) for m in MODALITIES], 1
                )
                recovered = self.generators[source](
                    context,
                    mask,
                    query_priors[source][rows] if query_priors is not None else None,
                )
                distance = 1 - F.cosine_similarity(
                    recovered, source_features[source].detach(), dim=-1
                )
                per_sample = per_sample.index_add(0, torch.where(rows)[0], distance)
        counts = selected.sum((1, 2)).clamp_min(1)
        return (per_sample / counts).mean()
