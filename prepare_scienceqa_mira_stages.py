"""Prepare the same caption -> KG -> subgraph evidence path for MIRA train/test.

Only ScienceQA train triples form the shared KG. At test time each question's
own unlabeled caption/hint triples may be added to retrieval, but no other
test question's triples or gold answer enter the corpus.
"""

import argparse
import getpass
import json
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scienceqa_full_pipeline import MODEL_ID, generate, parse_triples, retrieve, save_json
from scienceqa_gpt5_kg import gpt5_triplets
from smoke_scienceqa import choose_rows
from scienceqa_loader import load_scienceqa
from train_fusion_ablation import ROOT, image_from_row, select_train


def call_gpt5(key: str, prompt: str, base_url: str) -> str:
    for attempt in range(6):
        try:
            return gpt5_triplets(key, prompt, base_url, "gpt-5", True)
        except Exception as exc:
            message = str(exc)
            retryable = isinstance(exc, (requests.exceptions.ConnectionError,
                                         requests.exceptions.Timeout)) or any(
                code in message for code in ("HTTP 429", "HTTP 500", "HTTP 502",
                                              "HTTP 503", "HTTP 504", "timed out", "Timeout"))
            if not retryable or attempt == 5:
                raise
            pause = min(60, 2 ** attempt * 2)
            print(f"GPT-5 transient error; retrying in {pause}s", flush=True)
            time.sleep(pause)
    raise RuntimeError("Unreachable GPT-5 retry state")


def stage_record(row: dict, folder: Path, model, processor, reuse: Path | None,
                 *, kg_model: str, api_key: str, base_url: str,
                 cached_caption: str | None = None) -> list[tuple[str, str, str]]:
    folder.mkdir(parents=True, exist_ok=True)
    source = {k: v for k, v in row.items() if k not in
              {"image", "answer", "answer_index", "lecture", "solution"}}
    source["has_image"] = row["image"] is not None
    save_json(folder / "00_input.json", source)
    caption_file = folder / "01_caption.json"
    kg_file = folder / "02_kg.json"
    if kg_file.exists():
        try:
            saved = json.loads(kg_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = None  # A previous interrupted request left a partial file.
        if saved is not None:
            if saved.get("model") in (kg_model, None) and caption_file.exists():
                return [tuple(x) for x in saved["triples"]]
            raise RuntimeError(f"Existing KG model differs from requested {kg_model}: {kg_file}")

    if caption_file.exists():
        caption = json.loads(caption_file.read_text(encoding="utf-8"))["caption"]
    elif cached_caption is not None:
        caption = cached_caption
        save_json(caption_file, {"caption": caption, "model": MODEL_ID,
                                 "source": "local Qwen caption cache"})
    elif reuse is not None and (reuse / "01_caption.json").exists():
        reference_id = json.loads((reuse / "00_input.json").read_text(encoding="utf-8"))["id"]
        if reference_id != row["id"]:
            raise RuntimeError(f"Cached stage mismatch: {reference_id} vs {row['id']}")
        caption = json.loads((reuse / "01_caption.json").read_text(encoding="utf-8"))["caption"]
        save_json(caption_file, {"caption": caption, "model": MODEL_ID,
                                 "source": str(reuse)})
    else:
        image = image_from_row(row)
        caption_prompt = ("Describe only visible facts in this image relevant to the question. "
                          "Do not answer the question or use unseen facts.\nQuestion: " + row["question"])
        caption = generate(model, processor, caption_prompt, image, 128) if image is not None else ""
        save_json(caption_file, {"prompt": caption_prompt if image is not None else None,
                                 "caption": caption, "model": MODEL_ID})
    source_text = "\n".join(x for x in (caption, row.get("hint") or "") if x)
    kg_prompt = ("Extract up to 8 factual knowledge triples from the source text. "
                 "Use only stated facts; do not infer the answer from the question or choices. "
                 "Write one triple per line in exactly this format: subject | relation | object. "
                 "If no factual triples can be extracted, write NONE.\n\n"
                 f"Question (context only): {row['question']}\nSource text:\n{source_text or '(none)'}")
    if source_text and kg_model == "gpt-5":
        raw = call_gpt5(api_key, kg_prompt, base_url)
    elif source_text:
        raw = generate(model, processor, kg_prompt, max_new_tokens=180)
    else:
        raw = "NONE"
    triples = parse_triples(raw)
    save_json(kg_file, {"source_text": source_text, "prompt": kg_prompt,
                         "raw_text": raw, "triples": triples,
                         "model": kg_model if source_text else None,
                         "backend": "third-party chat-completions gateway" if kg_model == "gpt-5" and source_text else "local"})
    return triples


def ensure_caption(row: dict, folder: Path, model, processor,
                   cached_caption: str | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    source = {k: v for k, v in row.items() if k not in
              {"image", "answer", "answer_index", "lecture", "solution"}}
    source["has_image"] = row["image"] is not None
    save_json(folder / "00_input.json", source)
    path = folder / "01_caption.json"
    if path.exists():
        return
    if cached_caption is not None:
        save_json(path, {"caption": cached_caption, "model": MODEL_ID,
                         "source": "local Qwen caption cache"})
        return
    image = image_from_row(row)
    prompt = ("Describe only visible facts in this image relevant to the question. "
              "Do not answer the question or use unseen facts.\nQuestion: " + row["question"])
    caption = generate(model, processor, prompt, image, 128) if image is not None else ""
    save_json(path, {"prompt": prompt if image is not None else None,
                     "caption": caption, "model": MODEL_ID})


def caption_cache(*roots: Path) -> dict[str, str]:
    cache = {}
    for root in roots:
        if root is None:
            continue
        for folder in root.glob("**/01_caption.json"):
            input_file = folder.parent / "00_input.json"
            if input_file.exists():
                row_id = json.loads(input_file.read_text(encoding="utf-8"))["id"]
                cache[row_id] = json.loads(folder.read_text(encoding="utf-8"))["caption"]
    return cache


def retrieve_cached(global_kg, global_vectors, local_kg, query, encoder):
    """Same top-five/one-hop retrieval as retrieve(), with shared KG embedded once."""
    corpus = list(dict.fromkeys(global_kg + local_kg))
    if not corpus:
        return [], []
    global_set = set(global_kg)
    extra = [t for t in dict.fromkeys(local_kg) if t not in global_set]
    vectors = global_vectors
    if extra:
        extra_texts = [f"{s} {r} {o}" for s, r, o in extra]
        extra_vectors = encoder.encode(extra_texts, convert_to_tensor=True, normalize_embeddings=True)
        vectors = torch.cat((global_vectors, extra_vectors), dim=0)
    query_vector = encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = torch.matmul(vectors, query_vector).tolist()
    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
    seeds = [triple for triple, _ in ranked[:5]]
    adjacency = defaultdict(list)
    for triple in corpus:
        adjacency[triple[0]].append(triple)
        adjacency[triple[2]].append(triple)
    seen = set(seeds)
    queue = deque((node, 0) for s, _, o in seeds for node in (s, o))
    expanded = []
    while queue and len(expanded) < 8:
        node, depth = queue.popleft()
        if depth >= 1:
            continue
        for triple in adjacency[node]:
            if triple not in seen:
                seen.add(triple)
                expanded.append(triple)
                queue.append((triple[0], depth + 1))
                queue.append((triple[2], depth + 1))
            if len(expanded) >= 8:
                break
    return seeds, list(dict.fromkeys(seeds + expanded))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-count", type=int, default=None)
    ap.add_argument("--test-count", type=int, default=None)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "scienceqa_stages")
    ap.add_argument("--reuse-test", type=Path)
    ap.add_argument("--kg-model", choices=(MODEL_ID, "gpt-5"), default=MODEL_ID)
    ap.add_argument("--base-url", default="https://api2.aigcbest.top/v1")
    ap.add_argument("--kg-workers", type=int, default=8)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = select_train(args.train_count)
    test = (choose_rows(args.test_count) if args.test_count is not None else
            list(load_scienceqa("test", decode_images=True, include_references=True)))
    split = {"train_ids": [r["id"] for r in train], "test_ids": [r["id"] for r in test]}
    save_json(args.output / "split.json", split)
    if set(split["train_ids"]) & set(split["test_ids"]):
        raise RuntimeError("Train/test overlap")
    api_key = ""
    if args.kg_model == "gpt-5":
        api_key = getpass.getpass("GPT-5 gateway API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
        if not api_key:
            raise ValueError("GPT-5 gateway API key required on stdin")
    cached_captions = caption_cache(args.output, args.reuse_test)

    local = {}
    selected = [(part, number, row) for part, rows in (("train", train), ("test", test))
                for number, row in enumerate(rows, 1)]
    needs_qwen = args.kg_model == MODEL_ID or any(
        row["image"] is not None and row["id"] not in cached_captions and
        not (args.output / part / row["id"] / "01_caption.json").exists()
        for part, _, row in selected
    )
    if needs_qwen:
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None, local_files_only=True,
        ).eval()
    else:
        model = processor = None
    if args.kg_model == "gpt-5":
        for index, (part, number, row) in enumerate(selected, 1):
            folder = args.output / part / row["id"]
            ensure_caption(row, folder, model, processor, cached_captions.get(row["id"]))
            if index % 25 == 0 or index == len(selected):
                print(f"captions {index}/{len(selected)}", flush=True)
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with ThreadPoolExecutor(max_workers=args.kg_workers) as pool:
            futures = {}
            for part, number, row in selected:
                folder = args.output / part / row["id"]
                future = pool.submit(stage_record, row, folder, None, None, None,
                                     kg_model=args.kg_model, api_key=api_key,
                                     base_url=args.base_url)
                futures[future] = (part, number, row)
            completed = 0
            for future in as_completed(futures):
                part, number, row = futures[future]
                try:
                    local[row["id"]] = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
                completed += 1
                if completed % 25 == 0 or completed == len(selected):
                    print(f"GPT-5 KG {completed}/{len(selected)}", flush=True)
    else:
        for part, number, row in selected:
            folder = args.output / part / row["id"]
            reuse = (args.reuse_test / f"q{number:02d}" if args.reuse_test and part == "test" and
                     (args.reuse_test / f"q{number:02d}" / "00_input.json").exists() else None)
            triples = stage_record(row, folder, model, processor, reuse,
                                   kg_model=args.kg_model, api_key=api_key,
                                   base_url=args.base_url,
                                   cached_caption=cached_captions.get(row["id"]))
            local[row["id"]] = triples
            print(f"{part} {number}/{len(train) if part == 'train' else len(test)} {row['id']}: {len(triples)} KG triples", flush=True)
        del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    global_train_kg = list(dict.fromkeys(t for row in train for t in local[row["id"]]))
    save_json(args.output / "global_train_kg.json", {"model": args.kg_model,
               "triples": global_train_kg, "count": len(global_train_kg)})
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    global_texts = [f"{s} {r} {o}" for s, r, o in global_train_kg]
    global_vectors = encoder.encode(global_texts, convert_to_tensor=True, normalize_embeddings=True)
    for part, rows in (("train", train), ("test", test)):
        for row in rows:
            folder = args.output / part / row["id"]
            caption = json.loads((folder / "01_caption.json").read_text(encoding="utf-8"))["caption"]
            # Test examples may use their own unlabeled KG, never other test examples' KG.
            query = " ".join(x for x in (row["question"], caption, row.get("hint") or "") if x)
            seeds, subgraph = retrieve_cached(global_train_kg, global_vectors,
                                              local[row["id"]], query, encoder)
            clauses = [f"{s} {r.replace('_', ' ')} {o}" for s, r, o in subgraph]
            save_json(folder / "03_subgraph.json", {"query": query,
                      "seed_triples": seeds, "subgraph_triples": subgraph,
                      "corpus": "train KG plus own local KG"})
            save_json(folder / "04_evidence.json", {"caption": caption,
                      "hint": row.get("hint") or "", "evidence_clauses": clauses})
            print(f"retrieve {part} {row['id']}: {len(subgraph)} triples", flush=True)
    save_json(args.output / "summary.json", {"train_count": len(train),
              "test_count": len(test), "train_kg_triples": len(global_train_kg),
              "kg_model": args.kg_model, "caption_model": MODEL_ID,
              "kg_backend": "third-party chat-completions gateway" if args.kg_model == "gpt-5" else "local",
              "retriever": "sentence-transformers/all-MiniLM-L6-v2"})


if __name__ == "__main__":
    main()
