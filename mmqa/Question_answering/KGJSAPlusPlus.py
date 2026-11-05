import math
import torch
import torch.nn as nn


class KGJSAPlusPlus(nn.Module):
    """
    KG-JSA++ (Section F)
    - Projects routed HieraSlot memories (slot_d) → decoder dim (d_model)
    - Packs prefix memories (Eq.22)
    - Supports joint attention (Eq.24)
    """

    def __init__(
        self,
        d_model: int = 4096,   # LLaMA hidden size
        slot_d: int = 384,     # HieraSlot embedding size
        n_sink: int = 4,
        recent_window: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.slot_d = slot_d
        self.n_sink = n_sink
        self.recent_window = recent_window

        # Eq.(19): query projection
        self.W_Q = nn.Linear(d_model, d_model, bias=False)

        # Project slot memories to decoder space.
        self.W_mem_k = nn.Linear(slot_d, d_model, bias=False)
        self.W_mem_v = nn.Linear(slot_d, d_model, bias=False)
        self.decode_norm = nn.LayerNorm(d_model)
        self.decode_scale = nn.Parameter(torch.tensor(0.1))

    def build_prefix_memories(
        self,
        fusion_out: dict,
        route_out: dict | None,
        *,
        device: torch.device,
        K_T=None,
        V_T=None,
        K_Q=None,
        V_Q=None,
    ):
        """
        Implements Eq.(22):

        K_pref = [K_fus ; K_T ; K_Q ; K_mem]
        V_pref = [V_fus ; V_T ; V_Q ; V_mem]

        All memories are mapped to decoder dim d_model.
        """

        # ---- Fused multimodal prefix (already 4096)
        K_fus = torch.as_tensor(fusion_out["K_fus"], device=device)
        V_fus = torch.as_tensor(fusion_out["V_fus"], device=device)

        # ---- Optional text / instruction memories
        K_T = torch.tensor(K_T, device=device) if K_T is not None else K_fus[:0]
        V_T = torch.tensor(V_T, device=device) if V_T is not None else V_fus[:0]
        K_Q = torch.tensor(K_Q, device=device) if K_Q is not None else K_fus[:0]
        V_Q = torch.tensor(V_Q, device=device) if V_Q is not None else V_fus[:0]

        # ---- Routed HieraSlot memories
        if route_out is not None and "selected_keys" in route_out:
            K_mem = route_out["selected_keys"].to(device)  # [M, 384]
            V_mem = route_out["selected_values"].to(device)

            # Map routed memories to the decoder dimension.
            if K_mem.size(-1) != self.d_model:
                K_mem = self.W_mem_k(K_mem)
            if V_mem.size(-1) != self.d_model:
                V_mem = self.W_mem_v(V_mem)
        else:
            K_mem = K_fus[:0]
            V_mem = V_fus[:0]

        # ---- Eq.(22) concatenate all prefix memories
        K_pref = torch.cat([K_fus, K_T, K_Q, K_mem], dim=0)
        V_pref = torch.cat([V_fus, V_T, V_Q, V_mem], dim=0)

        return {
            "K_pref": K_pref,
            "V_pref": V_pref,
            "lens": {
                "fus": int(K_fus.size(0)),
                "T": int(K_T.size(0)),
                "Q": int(K_Q.size(0)),
                "mem": int(K_mem.size(0)),
                "total": int(K_pref.size(0)),
            }
        }

    def decode_prefix(self, h_t: torch.Tensor, prefix: dict, text_hidden: torch.Tensor):
        """Turn joint KGJSA attention into a decoder input embedding."""
        attended = self.joint_attention_step(
            h_t, prefix["K_pref"], prefix["V_pref"], text_hidden, text_hidden
        )
        return self.decode_scale * self.decode_norm(attended)

    def attend_sequence(self, hidden_states: torch.Tensor, prefix: dict):
        """Inject KGJSA memory at every decoder token in a hooked Llama layer."""
        q = self.W_Q(hidden_states.float())
        keys = prefix["K_pref"].float()
        values = prefix["V_pref"].float()
        scores = torch.matmul(q, keys.T) / math.sqrt(self.d_model)
        context = torch.matmul(torch.softmax(scores, dim=-1), values)
        update = self.decode_scale * self.decode_norm(context)
        return hidden_states + update.to(hidden_states.dtype)

    def joint_attention_step(
        self,
        h_t: torch.Tensor,      # [4096]
        K_pref: torch.Tensor,   # [M, 4096]
        V_pref: torch.Tensor,   # [M, 4096]
        K_self: torch.Tensor,   # cached decoder KV
        V_self: torch.Tensor,
    ):
        """
        Implements Eq.(24): joint attention update
        """
        d = self.d_model

        # Eq.(19)
        q_t = self.W_Q(h_t)  # [4096]

        # ---- Prefix attention
        S_pref = (q_t @ K_pref.T) / math.sqrt(d)
        A_pref = torch.softmax(S_pref, dim=0)
        o_pref = A_pref @ V_pref

        # ---- Sparse self-attention (simple recent window)
        T = K_self.size(0)
        start = max(0, T - self.recent_window)
        K_sel = K_self[start:T]
        V_sel = V_self[start:T]

        S_self = (q_t @ K_sel.T) / math.sqrt(d)
        A_self = torch.softmax(S_self, dim=0)
        o_self = A_self @ V_sel

        # ---- Eq.(24) output
        o_t = o_pref + o_self

        return o_t
