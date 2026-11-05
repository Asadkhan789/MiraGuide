# Hieraslot.py
"""
HieraSlot Encoder (Eq. 11–13)
Used after Global Multimodal Prefix
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384


# =========================
# Frozen sentence encoder φ(·)
# =========================
class FrozenSentenceEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts):
        return torch.tensor(
            self.model.encode(texts, normalize_embeddings=True),
            dtype=torch.float32,
            device=DEVICE
        )


# =========================
# HieraSlot Encoder
# =========================
class HieraSlotEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.phi = FrozenSentenceEncoder(EMB_MODEL_NAME)

        self.W_k = nn.Linear(EMB_DIM, EMB_DIM)
        self.W_v = nn.Linear(EMB_DIM, EMB_DIM)

        self.ln_k = nn.LayerNorm(EMB_DIM)
        self.ln_v = nn.LayerNorm(EMB_DIM)

        self.to(DEVICE)

    def summary(self, s, r, o):
        return f"{s} {r} {o}"

    def build_slots(self, triples):
        """
        triples: List[(s, r, o)]
        returns: hierarchical KV slots
        """

        key_texts = []
        val_texts = []
        slot_types = []

        for s, r, o in triples:
            # ---- Entity slot
            key_texts.append(s)
            val_texts.append(f"{r} → {o}")
            slot_types.append("entity")

            # ---- Relation slot
            key_texts.append(r)
            val_texts.append(self.summary(s, r, o))
            slot_types.append("relation")

        phi_k = self.phi.encode(key_texts)
        phi_v = self.phi.encode(val_texts)

        return self.project_slots(phi_k, phi_v, slot_types, triples)

    def project_slots(self, phi_k, phi_v, slot_types=None, triples=None):
        """Trainable slot projection from cached frozen sentence embeddings."""
        k_tilde = self.ln_k(self.W_k(phi_k))
        v_tilde = self.ln_v(self.W_v(phi_v))
        # simple placeholder evidence features (can be improved later)
        num_slots = k_tilde.size(0)
        features = torch.zeros(num_slots, 16, device=DEVICE)

        return {
            "keys": k_tilde,     # (2M, d)
            "values": v_tilde,   # (2M, d)
            "features": features,
            "meta": {
                "slot_type": slot_types or [],
                "triples": triples or []
            }
        }
