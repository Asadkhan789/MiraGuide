"""One-step gradient and generation check for the open-answer MIRA connector."""

import json

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_fusion_ablation import MODEL, ROOT
from train_mira_openqa import GlobalPrefixBuilder, HieraSlotEncoder, HieraSlotRouter, KGJSAPlusPlus


def main():
    device = torch.device("cuda")
    llama = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                  attn_implementation="sdpa").to(device).eval()
    llama.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    fusion = GlobalPrefixBuilder(prefix_len=64, load_encoders=False,
                                  hidden_size=llama.config.hidden_size).to(device)
    slots = HieraSlotEncoder().to(device)
    del slots.phi
    router = HieraSlotRouter(d_model=llama.config.hidden_size, d_key=384,
                             d_vis=llama.config.hidden_size, d_task=256, d_f=16,
                             top_k=8, epsilon=0.6).to(device)
    kgjsa = KGJSAPlusPlus(d_model=llama.config.hidden_size, slot_d=384).to(device)
    prompt = tokenizer("Question: What is shown?\nAnswer:", return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        hidden = llama.model(input_ids=prompt, use_cache=False).last_hidden_state[0].float()
    hidden = hidden.clone()
    fused = fusion.project_features(hidden[:96], torch.randn(1, 1536, device=device))
    projected = slots.project_slots(torch.randn(2, 384, device=device),
                                     torch.randn(2, 384, device=device),
                                     triples=[("image", "shows", "a test object")])
    routed = router(h_t=hidden[-1], v_t=fused["V_fus"].mean(0), slots=projected, task="QA")
    prefix = kgjsa.build_prefix_memories(fused, routed, device=device)
    target = tokenizer(" a test object", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat((prompt, target), dim=1)
    labels = full.clone()
    labels[:, :prompt.shape[1]] = -100
    hook = llama.model.layers[-1].register_forward_hook(
        lambda _module, _inputs, output: kgjsa.attend_sequence(output, prefix))
    try:
        logits = llama(input_ids=full, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
    finally:
        hook.remove()
    grad = sum(p.grad.float().norm().item() for module in (fusion, slots, router, kgjsa)
               for p in module.parameters() if p.grad is not None)
    if not torch.isfinite(loss) or grad <= 0:
        raise RuntimeError(f"Non-finite loss or zero gradient: loss={loss}, grad={grad}")
    print(json.dumps({"loss": float(loss.detach()), "gradient_norm_sum": grad,
                      "prefix_tokens": prefix["lens"]["total"]}), flush=True)


if __name__ == "__main__":
    main()
