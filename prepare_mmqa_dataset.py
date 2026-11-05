"""Prepare every labeled official MMQA train/dev row by default."""

import argparse
import gzip
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data" / "mmqa"


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def overlap(question, text):
    query = set(re.findall(r"[a-z0-9]+", question.lower()))
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query & words)


def choose_context(row, texts, tables, images):
    meta = row["metadata"]
    candidates = []
    for id_ in meta.get("text_doc_ids", []):
        doc = texts.get(id_)
        if doc:
            candidates.append((overlap(row["question"], doc["title"] + " " + doc["text"]),
                               "text", doc["title"] + ": " + doc["text"][:550]))
    table = tables.get(meta.get("table_id"))
    if table:
        headers = [h.get("column_name") or f"column{i}" for i, h in enumerate(table["table"]["header"])]
        for cells in table["table"]["table_rows"]:
            text = "; ".join(f"{header}: {cell.get('text','')}" for header, cell in zip(headers, cells)
                             if cell.get("text", "").strip())
            candidates.append((overlap(row["question"], text), "table",
                               f"{table['title']} / {table['table']['table_name']}: {text[:650]}"))
    ranked = sorted(candidates, key=lambda x: -x[0])
    snippets = [x[2] for x in ranked[:5]]
    image_docs = ([images[id_] for id_ in meta.get("image_doc_ids", []) if id_ in images]
                  if "image" in meta.get("modalities", []) else [])
    selected_image = max(image_docs, key=lambda d: overlap(row["question"], d["title"])) if image_docs else None
    return snippets, selected_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    args = parser.parse_args()
    texts = {x["id"]: x for x in read_jsonl_gz(ROOT / "MMQA_texts.jsonl.gz")}
    tables = {x["id"]: x for x in read_jsonl_gz(ROOT / "MMQA_tables.jsonl.gz")}
    images = {x["id"]: x for x in read_jsonl_gz(ROOT / "MMQA_images.jsonl.gz")}
    selected = {}
    for split, file, count in (("train", "MMQA_train.jsonl.gz", args.train_count),
                               ("val", "MMQA_dev.jsonl.gz", args.val_count)):
        rows = []
        for i, raw in enumerate(read_jsonl_gz(ROOT / file)):
            if count is not None and i == count:
                break
            if not raw.get("answers"):
                raise ValueError(f"Missing answer: {raw['qid']}")
            snippets, image = choose_context(raw, texts, tables, images)
            rows.append({"id": f"mmqa-{split}-{raw['qid']}", "question": raw["question"],
                         "answer": raw["answers"][0]["answer"],
                         "answers": [a["answer"] for a in raw["answers"]],
                         "modalities": raw["metadata"].get("modalities", []),
                         "context_snippets": snippets,
                         "image_title": image["title"] if image else "",
                         "image_name": image["path"] if image else "",
                         "image_path": str(ROOT / "images" / image["path"]) if image else ""})
        if count is not None and len(rows) != count:
            raise ValueError(f"Only {len(rows)} {split} rows found")
        selected[split] = rows
        with (ROOT / f"selected_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item) + "\n")
    if {x["id"].split("-")[-1] for x in selected["train"]} & {x["id"].split("-")[-1] for x in selected["val"]}:
        raise ValueError("Question ID overlap")
    summary = {"train": len(selected["train"]), "val": len(selected["val"]),
               "image_records": sum(bool(x["image_path"]) for split in selected.values() for x in split),
               "unique_images": len({x["image_path"] for split in selected.values() for x in split if x["image_path"]})}
    (ROOT / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
