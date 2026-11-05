"""Train MIRA fusion, HieraSlot, router and KGJSA on ScienceQA."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

from smoke_scienceqa import choose_rows
from scienceqa_loader import load_scienceqa
from train_fusion_ablation import MODEL, ROOT, STAGED, build_prompt, image_from_row, select_train

MCQ_TEMPERATURE = 1.0
MCQ_LENGTH_ALPHA = 1.0

sys.path.insert(0, str(ROOT / "mmqa" / "Question_answering"))
from fusion import GlobalPrefixBuilder  # noqa: E402
from Hieraslot import HieraSlotEncoder  # noqa: E402
from HieraSlotRouter import HieraSlotRouter  # noqa: E402
from KGJSAPlusPlus import KGJSAPlusPlus  # noqa: E402
from LatentCompressor import LatentCompressor  # noqa: E402


def triples_for(row: dict, test_number: int | None = None) -> list[tuple[str, str, str]]:
    if test_number is not None:
        path = STAGED / f"q{test_number:02d}" / "03_subgraph.json"
        saved = json.loads(path.read_text(encoding="utf-8"))["subgraph_triples"]
        if saved:
            return [tuple(map(str, triple)) for triple in saved]
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", row.get("hint") or "") if x.strip()]
    return [("ScienceQA question", "asks", row["question"][:180])] + [
        ("ScienceQA hint", "states", sentence[:180]) for sentence in sentences[:4]
    ]


def evidence_for(row: dict, test_number: int | None = None) -> str:
    if test_number is None:
        return ""
    folder = STAGED / f"q{test_number:02d}"
    source_id = json.loads((folder / "00_input.json").read_text(encoding="utf-8"))["id"]
    if source_id != row["id"]:
        raise RuntimeError(f"Staged evidence mismatch: {source_id} vs {row['id']}")
    clauses = json.loads((folder / "04_evidence_prompt.json").read_text(encoding="utf-8"))["evidence_clauses"]
    caption = json.loads((folder / "01_caption.json").read_text(encoding="utf-8")).get("caption", "")
    return "\n".join(x for x in (caption, *clauses) if x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-count", type=int, default=None)
    ap.add_argument("--test-count", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--router-epsilon", type=float, default=0.6)
    ap.add_argument("--no-epoch-checkpoints", action="store_true")
    ap.add_argument("--decoder-mode", choices=("prefix_token", "attention"), default="prefix_token")
    ap.add_argument("--generate-explanations", action="store_true")
    ap.add_argument("--compression", action="store_true",
                    help="Train a latent compressor and compare compressed and uncompressed KG-JSA memory")
    ap.add_argument("--compression-m0", type=int, default=16)
    ap.add_argument("--compression-r-qa", type=int, default=16)
    ap.add_argument("--compression-r-exp", type=int, default=8)
    ap.add_argument("--test-ids-file", type=Path,
                    help="JSON split file whose test_ids define the exact test subset")
    ap.add_argument("--checkpoint", type=Path,
                    help="Evaluate an existing connector without training")
    ap.add_argument("--benchmark-timing", action="store_true",
                    help="Measure off/on answer and explanation inference on the exact test IDs")
    ap.add_argument("--prepared", type=Path,
                    help="Caption/KG/subgraph stages prepared for both train and test")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "scienceqa_mira")
    args = ap.parse_args()
    if not 0.0 <= args.router_epsilon < 1.0:
        raise ValueError("--router-epsilon must be in [0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the CUDA-enabled pytorch conda environment")
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.benchmark_timing:
        (args.output / "timing.jsonl").write_text("", encoding="utf-8")
    train = select_train(args.train_count)
    if args.test_ids_file:
        requested = json.loads(args.test_ids_file.read_text(encoding="utf-8"))["test_ids"]
        if len(requested) != len(set(requested)) or (args.test_count is not None and len(requested) != args.test_count):
            raise ValueError("test_ids must contain unique IDs and match --test-count when set")
        wanted = set(requested)
        max_index = max(int(x.removeprefix("scienceqa-test-")) for x in requested)
        loaded = {x["id"]: x for x in load_scienceqa("test", limit=max_index + 1,
                    decode_images=True, include_references=True) if x["id"] in wanted}
        if set(loaded) != wanted:
            raise ValueError("Some requested ScienceQA test IDs were not found")
        test = [loaded[x] for x in requested]
    else:
        test = choose_rows(args.test_count) if args.test_count is not None else list(
            load_scienceqa("test", decode_images=True, include_references=True))
    rows = test if args.checkpoint else train + test
    if args.benchmark_timing and (not args.checkpoint or not args.generate_explanations):
        raise ValueError("--benchmark-timing requires --checkpoint and --generate-explanations")
    if {x["id"] for x in train} & {x["id"] for x in test}:
        raise RuntimeError("Train/test overlap")
    split = {"train_ids": [x["id"] for x in train], "test_ids": [x["id"] for x in test]}
    (args.output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    test_pos = {row["id"]: i for i, row in enumerate(test, 1)}
    if args.prepared:
        prepared_split = json.loads((args.prepared / "split.json").read_text(encoding="utf-8"))
        if (not set(split["train_ids"]).issubset(prepared_split["train_ids"]) or
                not set(split["test_ids"]).issubset(prepared_split["test_ids"])):
            raise RuntimeError("Prepared stages use a different train/test split")
        records = {}
        for part, selected in (("test", test),) if args.checkpoint else (("train", train), ("test", test)):
            for row in selected:
                folder = args.prepared / part / row["id"]
                source_id = json.loads((folder / "00_input.json").read_text(encoding="utf-8"))["id"]
                if source_id != row["id"]:
                    raise RuntimeError(f"Prepared evidence mismatch: {source_id} vs {row['id']}")
                triples = json.loads((folder / "03_subgraph.json").read_text(encoding="utf-8"))["subgraph_triples"]
                evidence = json.loads((folder / "04_evidence.json").read_text(encoding="utf-8"))
                records[row["id"]] = {"row": row, "triples": [tuple(x) for x in triples],
                                      "evidence": "\n".join(x for x in
                                       (evidence["caption"], *evidence["evidence_clauses"]) if x),
                                      "stage_dir": str(folder)}
    else:
        records = {row["id"]: {"row": row, "triples": triples_for(row, test_pos.get(row["id"])),
                                "evidence": evidence_for(row, test_pos.get(row["id"]))} for row in rows}
    device = torch.device("cuda")

    print("Caching frozen MiniLM slot features", flush=True)
    slots_model = HieraSlotEncoder().eval()
    for record in records.values():
        started = time.perf_counter()
        triples = record["triples"]
        key_texts, value_texts = [], []
        for s, r, o in triples:
            key_texts.extend((s, r))
            value_texts.extend((f"{r} → {o}", f"{s} {r} {o}"))
        record["phi_k"] = slots_model.phi.encode(key_texts).cpu()
        record["phi_v"] = slots_model.phi.encode(value_texts).cpu()
        record["slot_preprocess_s"] = time.perf_counter() - started
    del slots_model.phi
    torch.cuda.empty_cache()

    print("Caching frozen DINOv2 image features", flush=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant", use_fast=True)
    vision = AutoModel.from_pretrained("facebook/dinov2-giant", dtype=torch.float16).to(device).eval()
    with torch.inference_mode():
        for record in records.values():
            started = time.perf_counter()
            image = image_from_row(record["row"])
            record["has_image"] = image is not None
            if image is None:
                record["image_tokens"] = None
            else:
                inputs = processor(images=image, return_tensors="pt").to(device)
                record["image_tokens"] = vision(**inputs).last_hidden_state[:, 0].float().cpu()
            record["row"]["image"] = None  # release decoded image bytes after frozen encoding
            torch.cuda.synchronize()
            record["image_preprocess_s"] = time.perf_counter() - started
    del vision, processor
    torch.cuda.empty_cache()

    print("Loading frozen Llama and caching text features", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llama = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                  attn_implementation="sdpa").to(device).eval()
    llama.requires_grad_(False)
    llama.gradient_checkpointing_enable()
    embedding = llama.get_input_embeddings()
    letters = [tokenizer.encode(" " + chr(65+i), add_special_tokens=False)[-1] for i in range(26)]
    with torch.inference_mode():
        for record in records.values():
            started = time.perf_counter()
            prompt = build_prompt(record["row"], record["evidence"])
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=420).input_ids.to(device)
            record["ids"] = ids.cpu()
            torch.cuda.synchronize()
            record["direct_preprocess_s"] = time.perf_counter() - started
            record["text_hidden"] = llama.model(input_ids=ids, use_cache=False).last_hidden_state[0].float().cpu()
            torch.cuda.synchronize()
            record["text_preprocess_s"] = time.perf_counter() - started - record["direct_preprocess_s"]

    fusion = GlobalPrefixBuilder(prefix_len=64, load_encoders=False,
                                  hidden_size=llama.config.hidden_size).to(device)
    router = HieraSlotRouter(d_model=llama.config.hidden_size, d_key=384,
                             d_vis=llama.config.hidden_size, d_task=256, d_f=16,
                             top_k=8, epsilon=args.router_epsilon).to(device)
    kgjsa = KGJSAPlusPlus(d_model=llama.config.hidden_size, slot_d=384).to(device)
    compressor = (LatentCompressor(llama.config.hidden_size, args.compression_m0,
                                   args.compression_r_qa, args.compression_r_exp).to(device)
                  if args.compression else None)
    named_modules = {"fusion": fusion, "slots": slots_model, "router": router, "kgjsa": kgjsa}
    if compressor is not None:
        named_modules["compressor"] = compressor
    modules = tuple(named_modules.values())
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("router_epsilon", 0.0) != args.router_epsilon:
            raise ValueError("Checkpoint router threshold differs from --router-epsilon")
        if saved["split"]["train_ids"] != split["train_ids"] or not set(split["test_ids"]).issubset(saved["split"]["test_ids"]):
            raise ValueError("Checkpoint train split or test subset does not match")
        if ("compressor" in saved) != args.compression:
            raise ValueError("Checkpoint compression setting differs from --compression")
        for key, module in named_modules.items():
            module.load_state_dict(saved[key])
    else:
        parameters = [p for module in modules for p in module.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    def build_memory(record: dict, blank_image: bool = False,
                     compress: bool = False, task: str = "QA"):
        text_hidden = record["text_hidden"].to(device)
        image_tokens = None if blank_image else record["image_tokens"]
        if image_tokens is not None:
            image_tokens = image_tokens.to(device)
        fused = fusion.project_features(text_hidden[:96], image_tokens)
        slots = slots_model.project_slots(record["phi_k"].to(device),
                                           record["phi_v"].to(device), triples=record["triples"])
        routed = router(h_t=text_hidden[-1], v_t=fused["V_fus"].mean(0),
                        slots=slots, task=task)
        prefix = kgjsa.build_prefix_memories(fused, routed, device=device)
        uncompressed_count = prefix["lens"]["total"]
        if compress:
            if compressor is None:
                raise ValueError("Compression requested without a compressor")
            prefix = compressor(prefix, task=task)
        stage = {"slots": len(slots["keys"]), "routed": len(routed["selected_idx"]),
                 "prefix": prefix["lens"]["total"],
                 "uncompressed_prefix": uncompressed_count,
                 "image_in_fusion": image_tokens is not None}
        return prefix, text_hidden, stage

    def predict(record: dict, active: bool, *, blank_image: bool = False,
                train_mode: bool = False, compress: bool | None = None):
        row = record["row"]
        ids = record["ids"].to(device)
        prompt_embeds = embedding(ids)
        stage = {}
        handle = None
        if active:
            prefix, text_hidden, stage = build_memory(
                record, blank_image, compress=args.compression if compress is None else compress)
            if args.decoder_mode == "prefix_token":
                decode_token = kgjsa.decode_prefix(text_hidden[-1], prefix, text_hidden)
                embeds = torch.cat((decode_token.to(prompt_embeds.dtype).view(1, 1, -1), prompt_embeds), dim=1)
            else:
                embeds = prompt_embeds
                handle = llama.model.layers[-1].register_forward_hook(
                    lambda _module, _inputs, output: kgjsa.attend_sequence(output, prefix)
                )
        else:
            embeds = prompt_embeds
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
        try:
            logits = llama(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits[0, -1]
        finally:
            if handle is not None:
                handle.remove()
        choice_logits = logits[letters[:len(row["choices"])]]
        choice_token_count = 1
        choice_scores = torch.log_softmax(choice_logits.float() / MCQ_TEMPERATURE, dim=-1)
        choice_scores = choice_scores / (choice_token_count ** MCQ_LENGTH_ALPHA)
        if train_mode:
            target = torch.tensor([row["answer_index"]], device=device)
            return nn.functional.nll_loss(choice_scores[None], target), stage
        return int(choice_scores.argmax()), stage

    history = []
    for epoch in range(0 if args.checkpoint else args.epochs):
        for module in modules:
            module.train()
        for record in (records[row["id"]] for row in train):
            optimizer.zero_grad(set_to_none=True)
            loss, stage = predict(record, True, train_mode=True)
            loss.backward()
            grad_norms = {
                name: float(sum(p.grad.detach().float().norm().item()
                                for p in module.parameters() if p.grad is not None))
                for name, module in named_modules.items()
            }
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            item = {"epoch": epoch + 1, "id": record["row"]["id"],
                    "loss": float(loss.detach()), "grad_norms": grad_norms, **stage}
            history.append(item)
            print(json.dumps(item), flush=True)
        if not args.no_epoch_checkpoints:
            torch.save({"epoch": epoch + 1,
                        "router_epsilon": args.router_epsilon,
                        **{name: module.state_dict() for name, module in named_modules.items()},
                        "optimizer": optimizer.state_dict(), "split": split},
                       args.output / f"training_state_epoch{epoch + 1}.pt")
        (args.output / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not args.checkpoint:
        torch.save({name: module.cpu().state_dict() for name, module in named_modules.items()} | {
                "split": split, "seed": args.seed, "router_epsilon": args.router_epsilon},
                   args.output / "mira_connector.pt")
    for module in modules:
        module.to(device).eval()
    if not args.checkpoint:
        (args.output / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    results = []
    generated = []

    def explain(record: dict, choice_index: int, active: bool,
                token_times: list[float] | None = None, compress: bool | None = None) -> str:
        row = record["row"]
        letter = chr(65 + choice_index)
        choices = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(row["choices"]))
        messages = [
            {"role": "system", "content": "Explain the selected answer in one or two sentences. Output only the explanation."},
            {"role": "user", "content": f"Question: {row['question']}\nChoices:\n{choices}\n"
             f"Hint: {row.get('hint') or ''}\nEvidence: {record['evidence']}\n"
             f"Selected answer: {letter}. {row['choices'][choice_index]}"},
        ]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             return_tensors="pt").to(device)
        if ids.shape[1] > 450:
            ids = ids[:, -450:]
        handle = None
        if active:
            prefix, text_hidden, _ = build_memory(
                record, compress=args.compression if compress is None else compress, task="EXP")
            if args.decoder_mode == "attention":
                handle = llama.model.layers[-1].register_forward_hook(
                    lambda _module, _inputs, output: kgjsa.attend_sequence(output, prefix)
                )
                generate_kwargs = {"input_ids": ids,
                                   "attention_mask": torch.ones_like(ids)}
            else:
                token = kgjsa.decode_prefix(text_hidden[-1], prefix, text_hidden)
                embeds = torch.cat((token.to(embedding(ids).dtype).view(1, 1, -1), embedding(ids)), dim=1)
                generate_kwargs = {"inputs_embeds": embeds,
                                   "attention_mask": torch.ones(embeds.shape[:2], dtype=torch.long, device=device)}
        else:
            generate_kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if token_times is not None:
            class TokenTimer(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    torch.cuda.synchronize()
                    token_times.append(time.perf_counter())
                    return False
            generate_kwargs["stopping_criteria"] = [TokenTimer()]
        try:
            output = llama.generate(**generate_kwargs, max_new_tokens=80,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        finally:
            if handle is not None:
                handle.remove()
        generated_ids = output[0, ids.shape[1]:] if "input_ids" in generate_kwargs else output[0]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return raw.split("\n", 1)[0].strip()

    for row in test:
        record = records[row["id"]]
        timing = {}
        if args.benchmark_timing:
            with torch.no_grad():
                for arm, active in (("off", False), ("on", True)):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    answer, arm_stage = predict(record, active)
                    token_times = []
                    explanation = explain(record, answer, active, token_times)
                    torch.cuda.synchronize()
                    finished = time.perf_counter()
                    first = token_times[0] if token_times else finished
                    preproc = record["direct_preprocess_s"]
                    if active:
                        preproc += (record["slot_preprocess_s"] + record["image_preprocess_s"]
                                    + record["text_preprocess_s"])
                    timing[arm] = {"ttft_s": first - started,
                                   "decoding_ms_per_token": 1000 * (token_times[-1] - first) / (len(token_times) - 1) if len(token_times) > 1 else None,
                                   "generated_tokens": len(token_times),
                                   "cached_total_s": finished - started,
                                   "complete_total_s": finished - started + preproc,
                                   "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
                    if active:
                        on, stage, on_explanation = answer, arm_stage, explanation
                    else:
                        off, off_explanation = answer, explanation
                blank, _ = predict(record, True, blank_image=True)
        else:
            with torch.no_grad():
                off, _ = predict(record, False)
                on, stage = predict(record, True)
                blank, _ = predict(record, True, blank_image=True)
        if args.compression:
            with torch.no_grad():
                uncompressed, uncompressed_stage = predict(record, True, compress=False)
                if args.generate_explanations:
                    uncompressed_explanation = explain(record, uncompressed, True, compress=False)
        item = {"id": row["id"], "has_image": record["has_image"],
                "gold": row["answer_index"], "off": off, "on": on,
                "choices": row["choices"], "reference_answer": row["answer"],
                "stage_dir": record.get("stage_dir"),
                "on_blank_image": blank, "off_correct": off == row["answer_index"],
                "on_correct": on == row["answer_index"],
                "on_blank_image_correct": blank == row["answer_index"], **stage}
        results.append(item)
        if args.compression:
            item.update({"uncompressed": uncompressed,
                         "uncompressed_correct": uncompressed == row["answer_index"],
                         "uncompressed_prefix": uncompressed_stage["prefix"]})
        print(json.dumps(item), flush=True)
        if args.generate_explanations:
            if not args.benchmark_timing:
                with torch.no_grad():
                    off_explanation = explain(record, off, False)
                    on_explanation = explain(record, on, True)
            generated.append({"id": row["id"],
                              "reference_answer": row["answer"],
                              "reference_solution": row.get("solution") or "",
                              "off_answer": row["choices"][off], "off_explanation": off_explanation,
                              "off_correct": off == row["answer_index"],
                              "on_answer": row["choices"][on], "on_explanation": on_explanation,
                              "on_correct": on == row["answer_index"]})
            if args.compression:
                generated[-1].update({"uncompressed_answer": row["choices"][uncompressed],
                                      "uncompressed_explanation": uncompressed_explanation})
        if args.benchmark_timing:
            timing["id"] = row["id"]
            with (args.output / "timing.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(timing) + "\n")
    (args.output / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.generate_explanations:
        (args.output / "generated_explanations.json").write_text(
            json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"train_count": len(train), "test_count": len(test), "seed": args.seed,
               "off_correct": sum(x["off_correct"] for x in results),
               "on_correct": sum(x["on_correct"] for x in results),
               "on_blank_image_correct": sum(x["on_blank_image_correct"] for x in results),
               "image_count": sum(x["has_image"] for x in results),
               "decoder_mode": args.decoder_mode,
               "router_epsilon": args.router_epsilon, "learning_rate": args.learning_rate,
               "prepared_stages": str(args.prepared) if args.prepared else None,
               "generated_explanations": len(generated),
               "evaluation_checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "architecture": "Original fusion projections, HieraSlot, HieraSlotRouter and KGJSA"}
    if args.compression:
        summary.update({"compression": {"m0": args.compression_m0,
                                         "r_qa": args.compression_r_qa,
                                         "r_exp": args.compression_r_exp},
                        "uncompressed_correct": sum(x["uncompressed_correct"] for x in results)})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
