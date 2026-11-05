# HieraSlotRouter.py
"""
HieraSlotRouter — Section E: Dynamic Slot Selection

Implements Eq. (14):
g_p(t) = σ(q_t^T k_p + w_v^T v_t + w_task^T e_task + u_f^T φ(f_p) + b_g)

- Supports EXPLANATION task (task="EXP")
- Top-k routing + threshold ε
- Selected slots may be compressed downstream before KG-JSA attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as torch_F


class HieraSlotRouter(nn.Module):
    def __init__(
        self,
        d_model: int,   # decoder hidden size (e.g., 4096 for LLaMA-3.1-8B)
        d_key: int,     # slot key size (e.g., 384 from MiniLM)
        d_vis: int,     # visual context size (e.g., 4096 from fusion pooling)
        d_task: int,    # task embedding size
        d_f: int,       # slot feature size
        d_phi: int = 128,
        top_k: int = 8,
        epsilon: float = 0.6,
    ):
        super().__init__()
        self.top_k = top_k
        self.epsilon = epsilon

        # q_t = W_q h_t
        self.W_q = nn.Linear(d_model, d_key, bias=False)

        # φ(f_p) = MLP_f(f_p)
        self.mlp_f = nn.Sequential(
            nn.Linear(d_f, d_phi),
            nn.ReLU(),
            nn.Linear(d_phi, d_phi),
        )

        # Linear terms in Eq.14
        self.w_v = nn.Linear(d_vis, 1, bias=False)
        self.w_task = nn.Linear(d_task, 1, bias=False)
        self.u_f = nn.Linear(d_phi, 1, bias=False)
        self.b_g = nn.Parameter(torch.full((1,), 2.0))

        # Task embeddings
        self.task_emb = nn.ParameterDict({
            "QA":  nn.Parameter(torch.randn(d_task) * 0.02),
            "CAP": nn.Parameter(torch.randn(d_task) * 0.02),
            "EXP": nn.Parameter(torch.randn(d_task) * 0.02),  # EXPLANATION
        })

    def forward(
        self,
        h_t: torch.Tensor,   # [d_model]
        v_t: torch.Tensor,   # [d_vis]
        slots: dict,         # {"keys":[N,d_key], "values":[N,d_val], "features":[N,d_f]}
        task: str = "EXP",
    ):
        """
        Returns:
        {
          "selected_idx": LongTensor [M]
          "selected_keys": Tensor [M, d_key]
          "selected_values": Tensor [M, d_val]
          "scores": Tensor [N]
        }
        """
        assert task in self.task_emb, f"Unknown task {task}"

        K = slots["keys"]        # [N, d_key]
        V = slots["values"]      # [N, d_val]
        F = slots["features"]    # [N, d_f]

        if K.numel() == 0:
            return {
                "selected_idx": torch.empty(0, dtype=torch.long, device=K.device),
                "selected_keys": K,
                "selected_values": V,
                "scores": torch.empty(0, device=K.device),
            }

        # q_t = W_q h_t
        q_t = torch_F.normalize(self.W_q(h_t), dim=-1)  # stable query direction

        # Eq.14 terms
        term1 = 4.0 * torch.matmul(torch_F.normalize(K, dim=-1), q_t)  # [N]
        term2 = self.w_v(v_t).squeeze(0)         # scalar
        term3 = self.w_task(self.task_emb[task]).squeeze(0)  # scalar
        phi_f = self.mlp_f(F)                    # [N, d_phi]
        term4 = self.u_f(phi_f).squeeze(-1)      # [N]

        logits = term1 + term4 + term2 + term3 + self.b_g
        scores = torch.sigmoid(logits)           # g_p(t)

        # Top-k + threshold ε
        k = min(self.top_k, scores.size(0))
        top_vals, top_idx = torch.topk(scores, k=k)
        selected_idx = top_idx[top_vals > self.epsilon]

        return {
            "selected_idx": selected_idx,
            "selected_keys": K[selected_idx] * scores[selected_idx, None],
            "selected_values": V[selected_idx] * scores[selected_idx, None],
            "scores": scores,
        }


def build_visual_context_from_fusion(fusion_out: dict, d_vis: int, device: torch.device):
    """
    Builds v_t from fusion output.
    Uses mean-pooling over fused prefix if available.
    """
    if fusion_out is None:
        return torch.zeros(d_vis, device=device)

    # Preferred: pooled fused prefix
    if "V_fus" in fusion_out and isinstance(fusion_out["V_fus"], list):
        V = torch.tensor(fusion_out["V_fus"], device=device)
        return V.mean(dim=0)

    return torch.zeros(d_vis, device=device)
