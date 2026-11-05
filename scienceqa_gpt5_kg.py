"""Run ScienceQA KG construction with GPT-5, then retrieval and QA.

Reads one API key from stdin. The key is held in memory and never written to
the stage artifacts. Uses the official OpenAI Responses API by default.
"""

import argparse
import getpass
import json
import shutil
import sys
from pathlib import Path

import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scienceqa_full_pipeline import generate, parse_triples, retrieve, save_json
from smoke_scienceqa import choose_rows, extract_answer, extract_explanation, normalize


ROOT = Path(__file__).resolve().parent


def gpt5_triplets(api_key: str, prompt: str, base_url: str, model: str,
                  legacy_gateway: bool) -> str:
    if legacy_gateway:
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 2000, "temperature": 0, "reasoning_effort": "minimal"}
    else:
        url = base_url.rstrip("/") + "/responses"
        payload = {"model": model, "input": prompt, "reasoning": {"effort": "minimal"},
                   "max_output_tokens": 1000}
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(f"GPT-5 API HTTP {response.status_code}: {response.reason}")
    payload = response.json()
    if legacy_gateway:
        text = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    else:
        text = "\n".join(
            part.get("text", "")
            for item in payload.get("output", [])
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        ).strip()
    if not text:
        finish = payload.get("choices", [{}])[0].get("finish_reason") if legacy_gateway else payload.get("status")
        raise RuntimeError(f"GPT-5 returned no text (finish={finish})")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "scienceqa_full")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "scienceqa_gpt5_kg")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--legacy-gateway", action="store_true")
    parser.add_argument("--kg-only", action="store_true")
    args = parser.parse_args()
    api_key = (getpass.getpass("GPT-5 API key: ") if sys.stdin.isatty()
               else sys.stdin.readline().strip())
    if not api_key:
        raise ValueError("An API key must be provided on stdin")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = choose_rows(args.limit)
    manifest = []
    global_kg = []

    for number, row in enumerate(rows, 1):
        source = args.source / f"q{number:02d}"
        folder = args.output / f"q{number:02d}"
        folder.mkdir(exist_ok=True)
        for filename in ("00_input.json", "00_image.png", "01_caption.json"):
            if (source / filename).exists():
                shutil.copyfile(source / filename, folder / filename)
        caption = json.loads((folder / "01_caption.json").read_text(encoding="utf-8"))["caption"]
        evidence = "\n".join(x for x in (caption, row["hint"]) if x)
        prompt = (
            "Extract up to 8 factual knowledge triples from the source text. "
            "Use only stated facts, not the gold answer or unstated outside knowledge. "
            "Write one triple per line as subject | relation | object. "
            "If the source has no factual triples, write NONE.\n\n"
            f"Question (context only): {row['question']}\nSource text:\n{evidence or '(none)'}"
        )
        record = {"number": number, "id": row["id"], "has_image": row["image"] is not None,
                  "status": "running", "stages": ["caption"]}
        manifest.append(record)
        save_json(args.output / "manifest.json", manifest)
        try:
            raw = gpt5_triplets(api_key, prompt, args.base_url, args.model,
                                args.legacy_gateway) if evidence else "NONE"
            triples = parse_triples(raw)
            save_json(folder / "02_kg.json", {"model": args.model if evidence else None,
                                               "source_text": evidence, "prompt": prompt,
                                               "raw_text": raw, "triples": triples})
            global_kg.extend(triples)
            record["stages"].append("kg")
        except Exception as exc:
            record["status"] = f"kg_error: {type(exc).__name__}: {exc}"
        save_json(args.output / "manifest.json", manifest)
        print(f"kg q{number:02d}: {record['status']}", flush=True)
        if record["status"].startswith("kg_error") and number == 1:
            break

    global_kg = list(dict.fromkeys(global_kg))
    save_json(args.output / "global_kg.json", {"model": args.model, "triples": global_kg,
                                               "count": len(global_kg)})
    if args.kg_only or not global_kg:
        return

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    ).eval()
    for record in manifest:
        if record["status"] != "running":
            continue
        number = record["number"]
        row = rows[number - 1]
        folder = args.output / f"q{number:02d}"
        caption = json.loads((folder / "01_caption.json").read_text(encoding="utf-8"))["caption"]
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
                raw = generate(model, processor, prompt + "\nWrite both Explanation: and Answer: lines.",
                               row["image"], 180)
            answer = extract_answer(raw, row["choices"])
            save_json(folder / "05_answer.json", {"raw_text": raw,
                                                   "explanation": extract_explanation(raw),
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
        "kg_model": args.model, "qa_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "questions": len(rows), "completed": len(complete),
        "correct": sum(x["correct"] for x in complete), "global_kg_triples": len(global_kg),
        "mira_decoder_used": False,
    })


if __name__ == "__main__":
    main()
