"""Train MIRA connector on labeled MMQA or VQA-X open-answer train/val.

Frozen MiniLM, DINOv2 and Llama encode inputs. Fusion, HieraSlot, router and
KGJSA receive gradients from teacher-forced answer (and VQA-X explanation)
tokens. KGJSA is injected at the final Llama layer at every decoding step.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

from train_fusion_ablation import MODEL, ROOT, image_from_row

sys.path.insert(0, str(ROOT / "mmqa" / "Question_answering"))
from fusion import GlobalPrefixBuilder
from Hieraslot import HieraSlotEncoder
from HieraSlotRouter import HieraSlotRouter
from KGJSAPlusPlus import KGJSAPlusPlus
from LatentCompressor import LatentCompressor


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(dataset, split):
    path = ROOT / "data" / dataset / f"selected_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalize_answer(value):
    return " ".join(re.findall(r"[a-z0-9]+", answer_text(value).lower()))


def answer_text(value):
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value).strip()


def prompt_for(row, evidence, dataset):
    instruction = ("Answer the question using the supplied evidence. Give a short answer, "
                   "then one sentence explaining it. Use exactly 'Answer:' and 'Explanation:' labels.")
    text = f"{instruction}\nQuestion: {row['question']}\nEvidence: {evidence[:2600]}"
    return [{"role": "user", "content": text}]


def target_for(row, dataset):
    answer = answer_text(row["answer"])
    if dataset == "vqax" and row.get("explanations"):
        explanation = row["explanations"][0].strip()
        return f"Answer: {answer}\nExplanation: {explanation}"
    return f"Answer: {answer}"


def parse_output(raw):
    answer_part, _, explanation = raw.partition("Explanation:")
    answer = answer_part.strip().splitlines()[0].removeprefix("Answer:").strip() if answer_part.strip() else ""
    return answer, explanation.strip().splitlines()[0].strip() if explanation.strip() else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mmqa", "vqax"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--router-epsilon", type=float, default=0.6,
                        help="Minimum sigmoid gate score for each selected KG slot")
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--compression-m0", type=int, default=16)
    parser.add_argument("--compression-r-qa", type=int, default=16)
    parser.add_argument("--compression-r-exp", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path,
                        help="Evaluate saved connector weights without training again")
    args = parser.parse_args()
    if not 0.0 <= args.router_epsilon < 1.0:
        raise ValueError("--router-epsilon must be in [0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Llama 8B connector training")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train, val = read_rows(args.dataset, "train"), read_rows(args.dataset, "val")
    if args.train_limit is not None:
        train = train[:args.train_limit]
    if args.val_limit is not None:
        val = val[:args.val_limit]
    if not train or not val:
        raise ValueError("Both training and validation splits must be nonempty")
    ids = [r["id"] for r in train + val]
    if len(ids) != len(set(ids)):
        raise ValueError("Train/validation IDs overlap or repeat")
    split = {"train_ids": [r["id"] for r in train], "val_ids": [r["id"] for r in val]}
    prepared_split = read_json(args.prepared / "split.json")
    if (prepared_split["train_ids"][:len(train)] != split["train_ids"] or
            prepared_split["val_ids"][:len(val)] != split["val_ids"]):
        raise ValueError("Prepared stage split differs from the training split")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    records = {}
    for part, rows in (("val", val),) if args.checkpoint else (("train", train), ("val", val)):
        for row in rows:
            folder = args.prepared / part / row["id"]
            source = read_json(folder / "00_input.json")
            if source["id"] != row["id"]:
                raise ValueError(f"Stage mismatch: {row['id']}")
            triples = [tuple(x) for x in read_json(folder / "03_subgraph.json")["subgraph_triples"]]
            evidence = read_json(folder / "04_evidence.json")
            evidence_text = "\n".join(x for x in (
                *evidence.get("evidence_clauses", []), evidence.get("caption", ""),
                evidence.get("hint", "")) if x)
            records[row["id"]] = {"row": row, "triples": triples,
                                  "evidence": evidence_text, "stage_dir": str(folder)}

    print("Caching MiniLM slot embeddings", flush=True)
    slots = HieraSlotEncoder().eval()
    for record in records.values():
        triples = record["triples"] or [("question", "asks", record["row"]["question"])]
        record["triples"] = triples
        record["phi_k"] = slots.phi.encode([x for s, r, o in triples for x in (s, r)]).cpu()
        record["phi_v"] = slots.phi.encode([x for s, r, o in triples
                                             for x in (f"{r} → {o}", f"{s} {r} {o}")]).cpu()
    del slots.phi
    torch.cuda.empty_cache()

    print("Caching DINOv2 image features", flush=True)
    image_cache = args.output / "image_features"
    image_cache.mkdir(parents=True, exist_ok=True)
    needs_vision = any(r["row"].get("image_path") and
                       not (image_cache / f"{r['row']['id']}.pt").exists() for r in records.values())
    if needs_vision:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant", use_fast=True)
        vision = AutoModel.from_pretrained("facebook/dinov2-giant", dtype=torch.float16).to(device).eval()
    with torch.inference_mode():
        for i, record in enumerate(records.values(), 1):
            path = record["row"].get("image_path")
            cache_path = image_cache / f"{record['row']['id']}.pt"
            if cache_path.exists():
                record["image_tokens"] = torch.load(cache_path, map_location="cpu", weights_only=True)
            elif path:
                image = image_from_row({"image": {"path": path}})
                record["image_tokens"] = vision(**processor(images=image, return_tensors="pt").to(device)).last_hidden_state[:, 0].float().cpu()
                torch.save(record["image_tokens"], cache_path)
            else:
                record["image_tokens"] = None
            if i % 50 == 0:
                print(f"DINO {i}/{len(records)}", flush=True)
    if needs_vision:
        del processor, vision
    torch.cuda.empty_cache()

    print("Loading frozen Llama and caching prompt features", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llama = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                  attn_implementation="sdpa").to(device).eval()
    llama.requires_grad_(False)
    for i, record in enumerate(records.values(), 1):
        evidence_for_prompt = record["evidence"]
        while True:
            prompt = tokenizer.apply_chat_template(prompt_for(record["row"], evidence_for_prompt,
                                                               args.dataset), tokenize=False,
                                                   add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
            if ids.shape[1] <= 500:
                break
            if not evidence_for_prompt:
                raise ValueError(f"Question alone exceeds prompt limit: {record['row']['id']}")
            evidence_for_prompt = evidence_for_prompt[:int(len(evidence_for_prompt) * 0.8)]
        record["prompt_evidence_chars"] = len(evidence_for_prompt)
        ids = ids.to(device)
        record["ids"] = ids.cpu()
        with torch.inference_mode():
            hidden = llama.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        record["text_tokens"] = hidden[:96].float().cpu().clone()
        record["text_last"] = hidden[-1].float().cpu().clone()
        if i % 50 == 0:
            print(f"Llama features {i}/{len(records)}", flush=True)

    fusion = GlobalPrefixBuilder(prefix_len=64, load_encoders=False,
                                  hidden_size=llama.config.hidden_size).to(device)
    router = HieraSlotRouter(d_model=llama.config.hidden_size, d_key=384,
                             d_vis=llama.config.hidden_size, d_task=256, d_f=16,
                             top_k=8, epsilon=args.router_epsilon).to(device)
    kgjsa = KGJSAPlusPlus(d_model=llama.config.hidden_size, slot_d=384).to(device)
    compressor = (LatentCompressor(llama.config.hidden_size, args.compression_m0,
                                   args.compression_r_qa, args.compression_r_exp).to(device)
                  if args.compression else None)
    named_modules = {"fusion": fusion, "slots": slots, "router": router, "kgjsa": kgjsa}
    if compressor is not None:
        named_modules["compressor"] = compressor
    modules = tuple(named_modules.values())
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved["dataset"] != args.dataset or saved["split"] != split:
            raise ValueError("Checkpoint dataset or split differs from selected records")
        if saved.get("router_epsilon", 0.0) != args.router_epsilon:
            raise ValueError("Checkpoint router threshold differs from --router-epsilon")
        if ("compressor" in saved) != args.compression:
            raise ValueError("Checkpoint compression setting differs from --compression")
        for name, module in named_modules.items():
            module.load_state_dict(saved[name])
    else:
        parameters = [p for module in modules for p in module.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    def memory(record, *, task="QA", compress=None):
        text = record["text_tokens"].to(device)
        last = record["text_last"].to(device)
        image = record["image_tokens"]
        fused = fusion.project_features(text, image.to(device) if image is not None else None)
        projected = slots.project_slots(record["phi_k"].to(device), record["phi_v"].to(device),
                                         triples=record["triples"])
        routed = router(h_t=last, v_t=fused["V_fus"].mean(0), slots=projected, task=task)
        prefix = kgjsa.build_prefix_memories(fused, routed, device=device)
        if (args.compression if compress is None else compress):
            prefix = compressor(prefix, task=task)
        return prefix

    history = []
    for epoch in range(0 if args.checkpoint else args.epochs):
        for module in modules:
            module.train()
        for i, row in enumerate(train, 1):
            record = records[row["id"]]
            prefix = memory(record)
            prompt_ids = record["ids"].to(device)
            target = tokenizer(target_for(row, args.dataset), add_special_tokens=False,
                               return_tensors="pt").input_ids.to(device)
            target = torch.cat((target, torch.tensor([[tokenizer.eos_token_id]], device=device)), dim=1)
            if target.shape[1] > 80:
                target = target[:, :80]
            full = torch.cat((prompt_ids, target), dim=1)
            labels = full.clone()
            labels[:, :prompt_ids.shape[1]] = -100
            optimizer.zero_grad(set_to_none=True)
            hook = llama.model.layers[-1].register_forward_hook(
                lambda _module, _inputs, output: kgjsa.attend_sequence(output, prefix))
            try:
                logits = llama(input_ids=full, use_cache=False).logits
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       labels[:, 1:].reshape(-1), ignore_index=-100)
                loss.backward()
            finally:
                hook.remove()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            history.append({"epoch": epoch + 1, "id": row["id"],
                            "loss": float(loss.detach()),
                            "routed_slots": prefix["lens"]["mem"]})
            if i % 25 == 0:
                print(f"epoch {epoch+1} train {i}/{len(train)} loss {float(loss):.4f}", flush=True)
        (args.output / "train_history.json").write_text(json.dumps(history), encoding="utf-8")
        torch.save({"dataset": args.dataset, "split": split, "epoch": epoch + 1,
                    "router_epsilon": args.router_epsilon,
                    **{name: module.state_dict() for name, module in named_modules.items()}},
                   args.output / "mira_connector.pt")

    for module in modules:
        module.eval()
    class FirstSentenceStop(StoppingCriteria):
        def __init__(self, prompt_length):
            self.prompt_length = prompt_length

        def __call__(self, input_ids, scores, **kwargs):
            generated = tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
            return len(input_ids[0]) - self.prompt_length >= 8 and bool(
                re.search(r"[.!?](?:\s|$)", generated))

    results = []
    result_file = args.output / "validation_predictions.jsonl"
    result_file.write_text("", encoding="utf-8")
    with torch.inference_mode():
        for i, row in enumerate(val, 1):
            record = records[row["id"]]
            prefix = memory(record)
            ids = record["ids"].to(device)
            valid = [row["answer"], *row.get("answers", [])]
            item = {"id": row["id"], "question": row["question"],
                    "reference_answer": row["answer"], "reference_answers": valid,
                    "reference_explanations": row.get("explanations", []),
                    "stage_dir": record["stage_dir"],
                    "prompt_evidence_chars": record["prompt_evidence_chars"],
                    "routed_slots": prefix["lens"]["mem"]}
            for arm in (("off", "on", "uncompressed") if args.compression else ("off", "on")):
                arm_prefix = memory(record, compress=False) if arm == "uncompressed" else prefix
                hook = (llama.model.layers[-1].register_forward_hook(
                    lambda _module, _inputs, output: kgjsa.attend_sequence(output, arm_prefix))
                    if arm != "off" else None)
                try:
                    output = llama.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            max_new_tokens=48, do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id)
                finally:
                    if hook is not None:
                        hook.remove()
                raw = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
                answer, explanation = parse_output(raw)
                if args.dataset == "mmqa" and not explanation:
                    explanation_evidence = record["evidence"]
                    while True:
                        message = [{"role": "user", "content":
                                    "Explain the selected answer in one concise sentence using the evidence. "
                                    "Output only the explanation.\n"
                                    f"Question: {row['question']}\nSelected answer: {answer}\n"
                                    f"Evidence: {explanation_evidence}"}]
                        explanation_prompt = tokenizer.apply_chat_template(
                            message, tokenize=False, add_generation_prompt=True)
                        explanation_ids = tokenizer(explanation_prompt, return_tensors="pt",
                                                    truncation=False).input_ids
                        if explanation_ids.shape[1] <= 500:
                            break
                        if not explanation_evidence:
                            raise ValueError(f"Explanation prompt too long: {row['id']}")
                        explanation_evidence = explanation_evidence[:int(len(explanation_evidence) * 0.8)]
                    explanation_ids = explanation_ids.to(device)
                    explanation_prefix = (memory(record, task="EXP", compress=args.compression and arm == "on")
                                          if arm != "off" else None)
                    hook = (llama.model.layers[-1].register_forward_hook(
                        lambda _module, _inputs, output: kgjsa.attend_sequence(output, explanation_prefix))
                        if arm != "off" else None)
                    try:
                        explanation_output = llama.generate(
                            input_ids=explanation_ids, attention_mask=torch.ones_like(explanation_ids),
                            max_new_tokens=48, do_sample=False,
                            stopping_criteria=[FirstSentenceStop(explanation_ids.shape[1])],
                            pad_token_id=tokenizer.eos_token_id)
                    finally:
                        if hook is not None:
                            hook.remove()
                    explanation_raw = tokenizer.decode(
                        explanation_output[0, explanation_ids.shape[1]:],
                        skip_special_tokens=True).strip()
                    explanation = explanation_raw.splitlines()[0].removeprefix("Explanation:").strip()
                    item[f"{arm}_explanation_raw_generation"] = explanation_raw
                item[f"{arm}_answer"] = answer
                item[f"{arm}_explanation"] = explanation
                item[f"{arm}_raw_generation"] = raw
                item[f"{arm}_correct"] = (normalize_answer(answer) == normalize_answer(row["answer"])
                                          if args.dataset == "vqax" else normalize_answer(answer) in {
                                              normalize_answer(x) for x in valid})
                item[f"{arm}_any_gold_correct"] = normalize_answer(answer) in {
                    normalize_answer(x) for x in valid}
            results.append(item)
            with result_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"validation {i}/{len(val)} on correct {sum(x['on_correct'] for x in results)}", flush=True)
    summary = {"dataset": args.dataset, "train_count": len(train), "validation_count": len(val),
               "off_correct": sum(r["off_correct"] for r in results),
               "on_correct": sum(r["on_correct"] for r in results),
               "off_accuracy": sum(r["off_correct"] for r in results) / len(val),
               "on_accuracy": sum(r["on_correct"] for r in results) / len(val),
               "vqax_gold_rule": "majority answer exact match" if args.dataset == "vqax" else None,
               "epochs": args.epochs, "seed": args.seed, "learning_rate": args.learning_rate,
               "router_epsilon": args.router_epsilon, "router_top_k": 8,
               "validation_mean_routed_slots": sum(r["routed_slots"] for r in results) / len(val),
               "validation_zero_routed_slots": sum(r["routed_slots"] == 0 for r in results),
               "evaluation_checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "model": str(MODEL), "evaluation": "normalized exact match against available gold answers",
               "explanation_reference_available": args.dataset == "vqax",
               "architecture": "trainable fusion, HieraSlot, router and KGJSA final-layer attention; frozen encoders"}
    if args.compression:
        summary["compression"] = {"m0": args.compression_m0,
                                  "r_qa": args.compression_r_qa,
                                  "r_exp": args.compression_r_exp}
        summary["uncompressed_correct"] = sum(r["uncompressed_correct"] for r in results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
