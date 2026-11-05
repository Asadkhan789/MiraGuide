"""ScienceQA smoke run through caption, KG, subgraph, evidence, and QA stages.

Qwen supplies captioning, triplet extraction, and final QA until MIRA's Llama
weights are complete. The subgraph retrieval and evidence verbalization match
the existing MMQA stage logic (top five seeds, one-hop expansion, eight extras).
"""

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from smoke_scienceqa import choose_rows, extract_answer, extract_explanation, normalize


ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def messages(prompt: str, image: object = None) -> list[dict]:
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
    return [{"role": "user", "content": content}]


def generate(model, processor, prompt: str, image=None, max_new_tokens: int = 160) -> str:
    chat = messages(prompt, image)
    text = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image] if image is not None else None,
                       return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def parse_triples(raw: str) -> list[tuple[str, str, str]]:
    triples = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        line = line.removeprefix("(").removesuffix(")")
        parts = [p.strip().strip('"\'` ') for p in line.split("|")]
        if len(parts) == 3 and all(parts) and all(len(p) <= 120 for p in parts):
            triples.append(tuple(parts))
    return list(dict.fromkeys(triples))


def retrieve(kg: list[tuple[str, str, str]], query: str, encoder) -> tuple[list, list]:
    if not kg:
        return [], []
    texts = [f"{s} {r} {o}" for s, r, o in kg]
    triple_embs = encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    query_emb = encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, triple_embs)[0]
    ranked = sorted(zip(kg, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    seeds = [triple for triple, _ in ranked[:5]]

    adj = defaultdict(list)
    for triple in kg:
        adj[triple[0]].append(triple)
        adj[triple[2]].append(triple)
    seen = set(seeds)
    queue = deque((node, 0) for s, _, o in seeds for node in (s, o))
    expanded = []
    while queue and len(expanded) < 8:
        node, depth = queue.popleft()
        if depth >= 1:
            continue
        for triple in adj[node]:
            if triple not in seen:
                seen.add(triple)
                expanded.append(triple)
                queue.append((triple[0], depth + 1))
                queue.append((triple[2], depth + 1))
            if len(expanded) >= 8:
                break
    return seeds, list(dict.fromkeys(seeds + expanded))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "scienceqa_full")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = choose_rows(args.limit)
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None, local_files_only=True,
    ).eval()
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    manifest = []
    prepared = []

    # Stage 1: question-aware image captioning. Text rows retain their hint.
    for number, row in enumerate(rows, 1):
        folder = args.output / f"q{number:02d}"
        folder.mkdir(exist_ok=True)
        image = row["image"]
        source = {k: v for k, v in row.items() if k not in {"image", "answer", "answer_index", "lecture", "solution"}}
        source["has_image"] = image is not None
        save_json(folder / "00_input.json", source)
        if image is not None:
            image.convert("RGB").save(folder / "00_image.png")
        record = {"number": number, "id": row["id"], "has_image": image is not None,
                  "stages": [], "status": "running"}
        manifest.append(record)
        save_json(args.output / "manifest.json", manifest)
        try:
            caption_prompt = ("Describe only visible facts in this image relevant to the question. "
                              "Do not answer the question or use unseen facts.\nQuestion: " + row["question"])
            caption = generate(model, processor, caption_prompt, image, 128) if image is not None else ""
            save_json(folder / "01_caption.json", {"prompt": caption_prompt if image is not None else None,
                                                     "caption": caption, "hint": row["hint"]})
            record["stages"].append("caption")
            prepared.append((number, row, caption))
        except Exception as exc:
            record["status"] = f"caption_error: {type(exc).__name__}: {exc}"
        save_json(args.output / "manifest.json", manifest)
        print(f"caption q{number:02d}: {record['status']}", flush=True)

    # Stage 2: construct a KG from captions and hints, without gold answers.
    global_kg = []
    for number, row, caption in prepared:
        folder = args.output / f"q{number:02d}"
        record = manifest[number - 1]
        try:
            evidence = "\n".join(x for x in (caption, row["hint"]) if x)
            prompt = (
                "Extract up to 8 factual knowledge triples from the source text. "
                "Use only stated facts; do not infer the answer from the question or choices. "
                "Write one triple per line in exactly this format: subject | relation | object. "
                "If no factual triples can be extracted, write NONE.\n\n"
                f"Question (context only): {row['question']}\nSource text:\n{evidence or '(none)'}"
            )
            raw = generate(model, processor, prompt, max_new_tokens=180) if evidence else "NONE"
            triples = parse_triples(raw)
            save_json(folder / "02_kg.json", {"source_text": evidence, "prompt": prompt,
                                               "raw_text": raw, "triples": triples})
            global_kg.extend(triples)
            record["stages"].append("kg")
        except Exception as exc:
            record["status"] = f"kg_error: {type(exc).__name__}: {exc}"
        save_json(args.output / "manifest.json", manifest)
        print(f"kg q{number:02d}: {len(triples) if record['status'] == 'running' else record['status']}", flush=True)
    global_kg = list(dict.fromkeys(global_kg))
    save_json(args.output / "global_kg.json", {"triples": global_kg, "count": len(global_kg)})

    # Stages 3–4: MMQA-style retrieval and evidence-aware QA prompt, then answer.
    for number, row, caption in prepared:
        folder = args.output / f"q{number:02d}"
        record = manifest[number - 1]
        if record["status"] != "running":
            continue
        try:
            query = " ".join(x for x in (row["question"], caption, row["hint"]) if x)
            seeds, subgraph = retrieve(global_kg, query, encoder)
            save_json(folder / "03_subgraph.json", {"query": query, "seed_triples": seeds,
                                                    "subgraph_triples": subgraph})
            record["stages"].append("subgraph")
            clauses = [f"{s} {r.replace('_', ' ')} {o}" for s, r, o in subgraph]
            choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(row["choices"]))
            prompt = (
                "The following facts may help:\n" + "\n".join(clauses) + "\n\n"
                f"Question: {row['question']}\nChoices:\n{choices}\n"
                f"Hint: {row['hint'] or '(none)'}\n"
                "Use this exact two-part format:\n"
                "Explanation: <one or two sentences using the facts above and the image if present>\n"
                "Answer: <choice text>"
            )
            save_json(folder / "04_evidence_prompt.json", {"evidence_clauses": clauses, "prompt": prompt})
            record["stages"].append("evidence_prompt")
            raw = generate(model, processor, prompt, row["image"], 180)
            if not extract_explanation(raw):
                raw = generate(model, processor,
                               prompt + "\nYour previous reply omitted the explanation. "
                               "Write both Explanation: and Answer: lines.", row["image"], 180)
            answer = extract_answer(raw, row["choices"])
            explanation = extract_explanation(raw)
            save_json(folder / "05_answer.json", {"raw_text": raw, "explanation": explanation,
                                                   "answer": answer})
            correct = normalize(answer) == normalize(row["answer"])
            save_json(folder / "06_reference_check.json", {"reference_answer": row["answer"],
                                                            "reference_solution": row["solution"],
                                                            "correct": correct})
            record.update(status="complete", answer=answer, correct=correct)
            record["stages"].append("answer")
        except Exception as exc:
            record["status"] = f"answer_error: {type(exc).__name__}: {exc}"
        save_json(args.output / "manifest.json", manifest)
        print(f"answer q{number:02d}: {record['status']}", flush=True)

    complete = [x for x in manifest if x["status"] == "complete"]
    save_json(args.output / "summary.json", {
        "model": MODEL_ID, "questions": len(rows), "completed": len(complete),
        "correct": sum(x["correct"] for x in complete),
        "accuracy_on_completed": (sum(x["correct"] for x in complete) / len(complete)) if complete else None,
        "global_kg_triples": len(global_kg),
        "questions_without_local_triples": sum(
            not json.loads((args.output / f"q{x['number']:02d}" / "02_kg.json").read_text(encoding="utf-8"))["triples"]
            for x in manifest if "kg" in x["stages"]
        ),
        "image_questions": sum(x["has_image"] for x in manifest),
        "text_questions": sum(not x["has_image"] for x in manifest),
        "stages": ["caption", "kg", "subgraph", "evidence_prompt", "answer"],
        "mira_decoder_used": False,
    })


if __name__ == "__main__":
    main()
