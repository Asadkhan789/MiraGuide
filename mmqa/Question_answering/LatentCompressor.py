"""Trainable latent attention that bounds the KG-JSA prefix memory."""

import math

import torch
from torch import nn


class LatentCompressor(nn.Module):
    def __init__(self, d_model: int, m0: int = 16, r_qa: int = 16,
                 r_exp: int = 8, d_latent: int = 384):
        super().__init__()
        if not 1 <= r_qa <= m0 or not 1 <= r_exp <= m0:
            raise ValueError("Task budgets must be between 1 and m0")
        self.m0, self.r_qa, self.r_exp = m0, r_qa, r_exp
        self.norm = nn.LayerNorm(d_model)
        self.key = nn.Linear(d_model, d_latent, bias=False)
        self.value = nn.Linear(d_model, d_latent, bias=False)
        self.queries = nn.Parameter(torch.randn(m0, d_latent) * 0.02)
        self.out_k = nn.Linear(d_latent, d_model, bias=False)
        self.out_v = nn.Linear(d_latent, d_model, bias=False)

    def forward(self, prefix: dict, task: str = "QA") -> dict:
        if task not in ("QA", "EXP"):
            raise ValueError(f"Unsupported compression task: {task}")
        budget = self.r_qa if task == "QA" else self.r_exp
        keys = self.key(self.norm(prefix["K_pref"].float()))
        values = self.value(self.norm(prefix["V_pref"].float()))
        if keys.size(0) == 0:
            return prefix
        weights = torch.softmax(self.queries[:budget] @ keys.T / math.sqrt(keys.size(-1)), dim=-1)
        return {"K_pref": self.out_k(weights @ keys),
                "V_pref": self.out_v(weights @ values),
                "lens": {**prefix["lens"], "uncompressed_total": prefix["lens"]["total"],
                         "total": budget}}
