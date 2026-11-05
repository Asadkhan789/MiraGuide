"""Caption, GPT-5 KG, and retrieve evidence for MMQA or VQA-X train/val."""

import argparse
import getpass
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from prepare_scienceqa_mira_stages import ensure_caption, retrieve_cached, stage_record
from scienceqa_full_pipeline import MODEL_ID, save_json


ROOT = Path(__file__).resolve().parent


def load_rows(dataset: str, split: str):
    path = ROOT / "data" / dataset / f"selected_{split}.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        snippets = raw.get("context_snippets", [])
        rows.append({"id": raw["id"], "question": raw["question"],
                     "hint": "\n".join(snippets),
                     "image": {"path": raw["image_path"]} if raw.get("image_path") else None,
                     "dataset": dataset, "source_split": split})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mmqa", "vqax"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kg-workers", type=int, default=6)
    parser.add_argument("--base-url", default="https://api2.aigcbest.top/v1")
    args = parser.parse_args()
    train, val = load_rows(args.dataset, "train"), load_rows(args.dataset, "val")
    if not train or not val:
        raise ValueError("Both training and validation splits must be nonempty")
    if {r["id"] for r in train} & {r["id"] for r in val}:
        raise ValueError("Train/val ID overlap")
    for row in train + val:
        if row["image"] and not Path(row["image"]["path"]).exists():
            raise FileNotFoundError(row["image"]["path"])
    args.output.mkdir(parents=True, exist_ok=True)
    split = {"train_ids": [r["id"] for r in train], "val_ids": [r["id"] for r in val]}
    save_json(args.output / "split.json", split)
    key = getpass.getpass("GPT-5 gateway key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
    if not key:
        raise ValueError("GPT-5 gateway key required")
    selected = [(part, row) for part, group in (("train", train), ("val", val)) for row in group]
    pending_captions = [(part, row) for part, row in selected
                        if row["image"] and not (args.output / part / row["id"] / "01_caption.json").exists()]
    if pending_captions:
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None, local_files_only=True).eval()
    else:
        model = processor = None
    for i, (part, row) in enumerate(selected, 1):
        ensure_caption(row, args.output / part / row["id"], model, processor)
        if i % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("captions", i, "/", len(selected), flush=True)
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    local = {}
    with ThreadPoolExecutor(max_workers=args.kg_workers) as pool:
        futures = {pool.submit(stage_record, row, args.output / part / row["id"],
                               None, None, None, kg_model="gpt-5", api_key=key,
                               base_url=args.base_url): (part, row)
                   for part, row in selected}
        for i, future in enumerate(as_completed(futures), 1):
            part, row = futures[future]
            local[row["id"]] = future.result()
            if i % 50 == 0:
                print("KG", i, "/", len(selected), flush=True)
    global_kg = list(dict.fromkeys(t for row in train for t in local[row["id"]]))
    save_json(args.output / "global_train_kg.json", {"triples": global_kg, "count": len(global_kg),
              "requested_model": "gpt-5", "backend": "third-party gateway"})
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    global_texts = [f"{s} {r} {o}" for s, r, o in global_kg]
    global_vectors = encoder.encode(global_texts, convert_to_tensor=True, normalize_embeddings=True)
    for i, (part, row) in enumerate(selected, 1):
        folder = args.output / part / row["id"]
        caption = json.loads((folder / "01_caption.json").read_text(encoding="utf-8"))["caption"]
        query = " ".join(x for x in (row["question"], caption, row["hint"]) if x)
        seeds, subgraph = retrieve_cached(global_kg, global_vectors, local[row["id"]], query, encoder)
        clauses = [f"{s} {r.replace('_',' ')} {o}" for s, r, o in subgraph]
        save_json(folder / "03_subgraph.json", {"query": query, "seed_triples": seeds,
                  "subgraph_triples": subgraph, "corpus": "train KG plus own local KG"})
        save_json(folder / "04_evidence.json", {"caption": caption,
                  "hint": row["hint"], "evidence_clauses": clauses})
        if i % 50 == 0:
            print("retrieval", i, "/", len(selected), flush=True)
    save_json(args.output / "summary.json", {"dataset": args.dataset,
              "train_count": len(train), "val_count": len(val), "train_kg_triples": len(global_kg),
              "caption_model": MODEL_ID, "kg_model_requested": "gpt-5",
              "kg_backend": "third-party gateway",
              "retriever": "sentence-transformers/all-MiniLM-L6-v2"})


if __name__ == "__main__":
    main()
