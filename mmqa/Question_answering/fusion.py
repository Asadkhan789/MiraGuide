# fusion.py
"""
MIRA-Lite-MT — Section D
Global Multimodal Prefix
"""

import os
import glob
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoModel
)

TEXT_ENCODER_NAME = os.environ.get("MIRA_LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DINO_NAME = "facebook/dinov2-giant"


# =========================
# Utils
# =========================
def safe_join(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ; ".join(str(v) for v in x if v is not None)
    return str(x)


def build_T_comp(modality, img_caption, full_text, evidence_clauses):
    ev = safe_join(evidence_clauses)
    parts = [img_caption if modality == "image" else full_text, ev]
    return "\n".join(p.strip() for p in parts if p.strip())


def to_list(t):
    return t.detach().cpu().float().tolist()


# =========================
# Frozen DINOv2
# =========================
class FrozenDINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(DINO_NAME, use_fast=True)
        self.model = AutoModel.from_pretrained(
            DINO_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, img_path, dev):
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        return self.model(**inputs).last_hidden_state[0]  # (L_i, 1536)


# =========================
# Global Multimodal Prefix
# =========================
class GlobalPrefixBuilder(nn.Module):
    def __init__(self, prefix_len=64, *, load_encoders=True, hidden_size=4096):
        super().__init__()
        self.Lp = prefix_len

        # ---- Text encoder (frozen)
        if load_encoders:
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.text_lm = AutoModelForCausalLM.from_pretrained(
                TEXT_ENCODER_NAME,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            for p in self.text_lm.parameters():
                p.requires_grad = False
            self.text_lm.eval()
            self.d = self.text_lm.config.hidden_size
            self.img_encoder = FrozenDINOv2()
        else:
            self.tokenizer = None
            self.text_lm = None
            self.img_encoder = None
            self.d = hidden_size

        # ---- Projections
        self.Wt = nn.Linear(self.d, self.d)
        self.Wv = nn.Linear(1536, self.d)

        self.ln_text = nn.LayerNorm(self.d)
        self.ln_img = nn.LayerNorm(self.d)
        self.ln_fuse = nn.LayerNorm(self.d)

        self.fuse_proj = nn.Linear(self.d, self.d)

        self.WK = nn.Linear(self.d, self.d, bias=False)
        self.WV = nn.Linear(self.d, self.d, bias=False)

        # 🔑 dtype alignment
        if load_encoders:
            self.to(self.text_lm.dtype)

    def project_features(self, text_tokens: torch.Tensor, image_tokens: torch.Tensor | None = None):
        """Differentiable projection of frozen encoder features for connector training."""
        dtype = self.Wt.weight.dtype
        Et = self.ln_text(self.Wt(text_tokens.to(dtype)))
        if image_tokens is not None:
            Ei = self.ln_img(self.Wv(image_tokens.to(dtype)))
            image_budget = min(max(1, self.Lp // 4), Ei.size(0))
            E = torch.cat([Et[:self.Lp - image_budget], Ei[:image_budget]], dim=0)
        else:
            E = Et
        Ef = self._budget(self.ln_fuse(self.fuse_proj(E)))
        return {"E_fusion": Ef, "K_fus": self.WK(Ef), "V_fus": self.WV(Ef)}

    def _budget(self, x):
        if x.size(0) >= self.Lp:
            return x[:self.Lp]
        pad = torch.zeros(self.Lp - x.size(0), x.size(1),
                          device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    @torch.no_grad()
    def encode_text(self, T, dev):
        tok = self.tokenizer(T, return_tensors="pt",
                             truncation=True, max_length=512)
        tok = {k: v.to(dev) for k, v in tok.items()}
        return self.text_lm.model(**tok).last_hidden_state[0]

    @torch.no_grad()
    def forward(self, *, dev, modality, question, evidence_clauses,
                img_caption="", full_text="", img_path=None):

        Q = question or ""
        T_comp = build_T_comp(modality, img_caption, full_text, evidence_clauses)
        T = (Q.strip() + "\n" + T_comp.strip()).strip()

        # ---- Text stream
        Et = self.encode_text(T, dev)
        Et = self.ln_text(self.Wt(Et))

        # ---- Image stream (optional)
        if modality == "image":
            if not img_path or not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            Ei = self.img_encoder(img_path, dev)
            Ei = self.ln_img(self.Wv(Ei))
            image_budget = min(max(1, self.Lp // 4), Ei.size(0))
            E = torch.cat([Et[:self.Lp - image_budget], Ei[:image_budget]], dim=0)
        else:
            E = Et

        # ---- Fuse + budget
        Ef = self.ln_fuse(self.fuse_proj(E))
        Ef = self._budget(Ef)

        return {
            "Q": Q,
            "T_comp": T_comp,
            "T": T,
            "E_fusion": to_list(Ef),
            "K_fus": to_list(self.WK(Ef)),
            "V_fus": to_list(self.WV(Ef)),
        }
