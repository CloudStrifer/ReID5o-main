"""MFCRAF: one encoding path for training, evaluation and interactive retrieval."""

import torch
from torch import nn
from torch.nn import functional as F
from .clip_model import build_CLIP_from_openai_pretrained
from .spa_lora import StateConditionedPartAwareTransformer
from .missing_aware_encoding import (
    MissingAwareEncoder,
    ModalityDropout,
    ConsistencyConstraint,
)
from .cross_modal_completion import CrossModalCompletionModule, MODALITIES
from .reliability_adaptive_fusion import ReliabilityAdaptiveFusion
from .objectives import compute_sdm

INPUT_KEYS = ("rgbs", "nirs", "cps", "sks", "caption_ids")
ARCHITECTURE_VERSION = "mfcraf-v2"


class TextBranch(nn.Module):
    def __init__(self, base):
        super().__init__()
        for key in (
            "token_embedding",
            "positional_embedding",
            "transformer",
            "ln_final",
            "text_projection",
        ):
            setattr(self, key, getattr(base, key))

    def forward(self, text):
        x = self.token_embedding(text) + self.positional_embedding[: text.shape[1]]
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        return self.ln_final(x) @ self.text_projection


class ReID5oModel(nn.Module):
    def __init__(self, args, num_classes=11003, backbone=None):
        super().__init__()
        self.args, self.num_classes = args, num_classes
        base, cfg = backbone or build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = cfg["embed_dim"]
        self.visual, self.clip_text_encoder = base.visual, TextBranch(base)
        if args.text_length > base.positional_embedding.shape[0]:
            raise ValueError("text_length exceeds the pretrained CLIP context length")
        self.visual.requires_grad_(
            getattr(args, "freeze_vision_encoder", True) is False
        )
        self.use_spa_lora = getattr(args, "add_lora", True)
        if self.use_spa_lora:
            self.visual.transformer = StateConditionedPartAwareTransformer(
                self.visual.transformer,
                args.lora_r,
                getattr(args, "spa_experts", 4),
                getattr(args, "spa_parts", 3),
                args.lora_layers,
            )
        if getattr(args, "freeze_text_encoder", False):
            self.clip_text_encoder.requires_grad_(False)
        self.use_missing_aware = getattr(args, "use_missing_aware", True)
        self.use_cross_modal_completion = getattr(
            args, "use_cross_modal_completion", True
        )
        self.use_reliability_fusion = getattr(args, "use_reliability_fusion", True)
        self.use_completion_inference = getattr(args, "use_completion_inference", True)
        self.use_reliability_fusion_inference = getattr(
            args, "use_reliability_fusion_inference", True
        )
        if "ORBench" in args.dataset_name and not getattr(
            args, "force_missing_nir_cp", False
        ):
            active = [True] * 5
        else:
            active = [True, False, False, True, True]
        self.register_buffer(
            "active_modalities", torch.tensor(active, dtype=torch.bool)
        )
        h, w = args.img_size
        kh, kw = self.visual.conv1.kernel_size
        sh, sw = self.visual.conv1.stride
        self.grid = ((h - kh) // sh + 1, (w - kw) // sw + 1)
        self.missing_aware_encoder = MissingAwareEncoder(
            self.embed_dim,
            vision_num_tokens=1 + self.grid[0] * self.grid[1],
            text_num_tokens=args.text_length,
        )
        self.modality_dropout = ModalityDropout(
            min_keep=getattr(args, "min_modalities_keep", 1),
            dropout_prob=args.modality_dropout_prob,
            keep_rgb_prob=args.keep_rgb_prob,
        )
        self.consistency_constraint = ConsistencyConstraint()
        self.completion_module = CrossModalCompletionModule(
            self.embed_dim,
            num_heads=args.completion_num_heads,
            num_layers=args.completion_num_layers,
            dropout=args.completion_dropout,
        )
        self.reliability_fusion = ReliabilityAdaptiveFusion(
            self.embed_dim,
            hidden_dim=args.reliability_hidden_dim,
            temperature=getattr(args, "fusion_temperature", 1.0),
            entropy_target=getattr(args, "fusion_entropy_target", 0.3),
            margin=getattr(args, "fusion_uncertainty_margin", 0.5),
            use_quality_indicators=getattr(args, "use_quality_indicators", True),
        )
        self.fallback_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)
        self.logit_scale = 1 / args.temperature
        if args.loss_names not in ("mm_sdm+id", "sdm+id"):
            raise ValueError(
                "MFCRAF v2 implements the paper objective: --loss_names mm_sdm+id"
            )

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _availability(self, inputs, available=None):
        sample = next(
            (inputs[k] for k in INPUT_KEYS if k in inputs and inputs[k] is not None),
            None,
        )
        if sample is None:
            raise ValueError("At least one input tensor is required")
        b, device = len(sample), sample.device
        supplied = torch.tensor(
            [k in inputs and inputs[k] is not None for k in INPUT_KEYS], device=device
        )
        if available is None:
            available = inputs.get("modality_available")
        if available is None:
            available = supplied[None].expand(b, -1)
        if isinstance(available, dict):
            available = torch.stack(
                [
                    torch.as_tensor(available.get(m, False), device=device).expand(b)
                    for m in MODALITIES
                ],
                1,
            )
        available = torch.as_tensor(available, dtype=torch.bool, device=device)
        if available.ndim == 1:
            available = available[None].expand(b, -1)
        if available.shape != (b, 5):
            raise ValueError("Availability must be [B,5]")
        available = available & supplied[None] & self.active_modalities
        if not available.any(1).all():
            raise ValueError("Each sample needs a valid observed modality")
        return available

    def _visual_tokens(self, image, state):
        x = self.visual.conv1(image.to(self.dtype))
        grid = x.shape[-2:]
        x = x.flatten(2).permute(0, 2, 1)
        cls = self.visual.class_embedding.to(x.dtype).expand(len(x), 1, -1)
        x = torch.cat([cls, x], 1) + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x).permute(1, 0, 2)
        x = (
            self.visual.transformer(x, state, grid)
            if self.use_spa_lora
            else self.visual.transformer(x)
        )
        return self.visual.ln_post(x.permute(1, 0, 2)) @ self.visual.proj

    def encode_multimodal(
        self, inputs, modality_mask=None, available=None, return_details=False
    ):
        available = self._availability(inputs, available)
        observed = (
            available
            if modality_mask is None
            else torch.as_tensor(
                modality_mask, device=available.device, dtype=torch.bool
            )
        )
        if observed.ndim == 1:
            observed = observed[None].expand(len(available), -1)
        if observed.shape != available.shape or (observed & ~available).any():
            raise ValueError("Observed mask must be a subset of real availability")
        if not observed.any(1).all():
            raise ValueError("A sample has no observed modality")
        features, missing_priors = {}, {}
        for k, m in enumerate(MODALITIES):
            n = self.args.text_length if k == 4 else 1 + self.grid[0] * self.grid[1]
            raw = torch.zeros(
                len(available),
                n,
                self.embed_dim,
                device=available.device,
                dtype=self.dtype,
            )
            indices = torch.where(observed[:, k])[0]
            if len(indices):
                data = inputs[INPUT_KEYS[k]][indices]
                encoded = (
                    self.clip_text_encoder(data)
                    if k == 4
                    else self._visual_tokens(data, observed[indices])
                )
                raw = raw.index_copy(0, indices, encoded)
            if self.use_missing_aware:
                tokens = self.missing_aware_encoder(raw, k, observed[:, k])
                prior_tokens = self.missing_aware_encoder(
                    None, k, torch.zeros_like(observed[:, k])
                )
                missing_priors[m] = prior_tokens.mean(1)
            else:
                tokens = raw
                missing_priors[m] = raw[:, 0] * 0
            if k == 4:
                eot = (
                    inputs["caption_ids"].argmax(-1)
                    if "caption_ids" in inputs
                    else torch.zeros(len(raw), device=raw.device, dtype=torch.long)
                )
                eot_feature = tokens[
                    torch.arange(len(tokens), device=tokens.device), eot
                ]
                pooled = torch.where(observed[:, k, None], eot_feature, tokens.mean(1))
            else:
                pooled = tokens[:, 0]
            features[m] = pooled
        use_completion = self.use_cross_modal_completion and (
            self.training or self.use_completion_inference
        )
        if use_completion:
            fusion_features, predictions = self.completion_module.complete(
                features, observed, self.active_modalities, missing_priors
            )
            valid = self.active_modalities[None].expand_as(observed)
        else:
            fusion_features, predictions = features, {}
            valid = observed
        generated = valid & ~observed
        use_fusion = self.use_reliability_fusion and (
            self.training or self.use_reliability_fusion_inference
        )
        if use_fusion:
            fused, info = self.reliability_fusion(fusion_features, valid, generated)
        else:
            stacked = torch.stack(
                [
                    fusion_features.get(m, torch.zeros_like(features["RGB"]))
                    for m in MODALITIES
                ],
                1,
            )
            weights = valid.to(stacked.dtype) / valid.sum(1, keepdim=True)
            fused = self.fallback_projection((weights[:, :, None] * stacked).sum(1))
            zero = fused.sum() * 0
            info = {
                "fusion_weights": weights,
                "sparsity_loss": zero,
                "uncertainty_loss": zero,
            }
        output = F.normalize(fused.float(), dim=-1)
        details = {
            **info,
            "pooled_features": features,
            "predictions": predictions,
            "missing_priors": missing_priors,
            "observed_mask": observed,
            "generated_mask": generated,
            "valid_mask": valid,
        }
        return (output, details) if return_details else output

    def forward(self, batch, use_modality_dropout=None, current_epoch=None):
        available = self._availability(batch)
        warmup = current_epoch is not None and current_epoch <= getattr(
            self.args, "modality_dropout_warmup_epochs", 5
        )
        drop = self.training and self.use_missing_aware and not warmup
        if use_modality_dropout is not None:
            drop = bool(use_modality_dropout)
        observed = batch.get("modality_mask")
        if observed is None:
            observed = self.modality_dropout(available, training=drop)
        query, info = self.encode_multimodal(batch, observed, available, True)
        labels = batch["pids"]
        zero = query.sum() * 0
        losses = {
            "id_loss": F.cross_entropy(self.classifier(query), labels)
            * self.args.id_loss_weight,
            "sdm_loss": zero,
        }
        rgb_rows = available[:, 0]
        if rgb_rows.any():
            gallery_inputs = {
                k: v[rgb_rows]
                for k, v in batch.items()
                if torch.is_tensor(v) and k in INPUT_KEYS
            }
            rgb_mask = torch.zeros(
                int(rgb_rows.sum()), 5, dtype=torch.bool, device=query.device
            )
            rgb_mask[:, 0] = True
            gallery = self.encode_multimodal(gallery_inputs, rgb_mask, rgb_mask)
            losses["sdm_loss"] = compute_sdm(
                gallery, query[rgb_rows], labels[rgb_rows], self.logit_scale
            )
            losses["id_loss"] = (
                losses["id_loss"]
                + F.cross_entropy(self.classifier(gallery), labels[rgb_rows])
                * self.args.id_loss_weight
            ) / 2
        if self.use_missing_aware or self.use_cross_modal_completion:
            with torch.no_grad():
                teacher, full = self.encode_multimodal(
                    batch, available, available, True
                )
            losses["consistency_loss"] = (
                self.consistency_constraint(teacher, query)
                * self.args.consistency_loss_weight
                if self.use_missing_aware
                else zero
            )
            if self.use_cross_modal_completion:
                losses["completion_recon_loss"] = (
                    self.completion_module.reconstruction_loss(
                        info["predictions"],
                        full["pooled_features"],
                        available & ~observed,
                    )
                    * self.args.completion_recon_loss_weight
                )
                if self.args.completion_cycle_loss_weight:
                    losses["completion_cycle_loss"] = (
                        self.completion_module.cycle_loss(
                            full["pooled_features"],
                            available,
                            self.active_modalities,
                            info["missing_priors"],
                            getattr(self.args, "cycle_pairs_per_step", 3),
                        )
                        * self.args.completion_cycle_loss_weight
                    )
                else:
                    losses["completion_cycle_loss"] = zero
        if self.use_reliability_fusion:
            losses["fusion_sparsity_loss"] = (
                info["sparsity_loss"] * self.args.fusion_sparsity_weight
            )
            losses["fusion_uncertainty_loss"] = (
                info["uncertainty_loss"] * self.args.fusion_uncertainty_weight
            )
        return {
            **losses,
            "features": query,
            "modality_mask": observed,
            "fusion_weights": info["fusion_weights"],
            "generated_mask": info["generated_mask"],
        }


def build_model(args, num_classes=11003, backbone=None):
    return ReID5oModel(args, num_classes, backbone)
