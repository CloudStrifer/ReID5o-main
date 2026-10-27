"""State-conditioned part-aware LoRA on the visual MLP (paper equations 3--5)."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class SPALoRALinear(nn.Module):
    def __init__(self, linear, rank=4, experts=4, parts=3, num_modalities=5):
        super().__init__()
        if min(rank, experts, parts) < 1:
            raise ValueError("rank, experts and parts must be positive")
        self.base = linear
        self.base.requires_grad_(False)
        self.parts, self.experts, self.rank = parts, experts, rank
        self.lora_A = nn.Parameter(
            torch.empty(parts, experts, rank, linear.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(parts, experts, linear.out_features, rank)
        )
        self.lora_router = nn.Parameter(torch.zeros(parts, experts, num_modalities))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_router, std=0.02)
        self.scaling = 1.0 / rank

    def routing_weights(self, state):
        return torch.einsum(
            "bk,pek->bpe", state.to(self.lora_router.dtype), self.lora_router
        ).softmax(-1)

    def forward(self, x, state, grid):
        # x: [batch,1+height*width,channels], row-major visual patches.
        h, w = grid
        if x.shape[1] != 1 + h * w or self.parts > h:
            raise ValueError("Invalid spatial grid or too many parts")
        gates = self.routing_weights(state)
        patch_parts = (
            torch.arange(h, device=x.device).repeat_interleave(w) * self.parts // h
        )
        delta = torch.zeros(
            *x.shape[:2], self.base.out_features, device=x.device, dtype=x.dtype
        )
        cls_deltas = []
        for p in range(self.parts):
            indices = torch.where(patch_parts == p)[0] + 1
            selected = torch.cat([x[:, :1], x[:, indices]], dim=1)
            low = torch.einsum("bli,eri->bler", selected, self.lora_A[p])
            outputs = torch.einsum("bler,eor->bleo", low, self.lora_B[p])
            mixed = torch.einsum("be,bleo->blo", gates[:, p], outputs) * self.scaling
            delta[:, indices] = mixed[:, 1:]
            cls_deltas.append(mixed[:, :1])
        delta[:, :1] = torch.stack(cls_deltas).mean(0)
        return self.base(x) + delta


class StateConditionedPartAwareTransformer(nn.Module):
    def __init__(self, transformer, rank=4, experts=4, parts=3, lora_layers=2):
        super().__init__()
        self.resblocks = transformer.resblocks
        if not 1 <= lora_layers <= len(self.resblocks):
            raise ValueError("Invalid lora_layers")
        for block in list(self.resblocks)[-lora_layers:]:
            block.mlp.c_fc = SPALoRALinear(block.mlp.c_fc, rank, experts, parts)
            block.mlp.c_proj = SPALoRALinear(block.mlp.c_proj, rank, experts, parts)

    def forward(self, x, state, grid):
        for block in self.resblocks:
            if isinstance(block.mlp.c_fc, SPALoRALinear):
                x = x + block.attention(block.ln_1(x))
                y = block.ln_2(x).permute(1, 0, 2)
                y = block.mlp.c_fc(y, state, grid)
                y = block.mlp.gelu(y)
                y = block.mlp.c_proj(y, state, grid)
                x = x + y.permute(1, 0, 2)
            else:
                x = block(x)
        return x
