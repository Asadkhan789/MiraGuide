"""Prepare every labeled VQA-X train/val row and its COCO image by default."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent / "data" / "vqax"


def rows(split: str, count: int | None):
    data = json.loads((ROOT / f"vqaX_{split}.json").read_text(encoding="utf-8"))
    selected = list(data.items())[:count]
    if count is not None and len(selected) != count:
        raise ValueError(f"Only {len(selected)} {split} records available")
    result = []
    for id_, row in selected:
        answers = [x["answer"].strip() for x in row["answers"] if x.get("answer", "").strip()]
        if not answers:
            raise ValueError(f"Missing answer for {id_}")
        answer = Counter(answers).most_common(1)[0][0]
        image = ROOT / "images" / split / row["image_name"]
        result.append({"id": f"vqax-{split}-{id_}", "question": row["question"],
                       "answer": answer, "answers": answers,
                       "explanations": row.get("explanation") or [],
                       "image_path": str(image), "image_name": row["image_name"]})
    return result


def fetch(item):
    path = Path(item["image_path"])
    if path.exists() and path.stat().st_size > 1000:
        return item["id"], "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    split = "train2014" if "train2014" in path.name else "val2014"
    url = f"http://images.cocodataset.org/{split}/{path.name}"
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            if not response.headers.get("Content-Type", "").startswith("image/"):
                raise ValueError("Response is not an image")
            part = path.with_suffix(path.suffix + ".part")
            part.write_bytes(response.content)
            part.replace(path)
            return item["id"], "downloaded"
        except (requests.RequestException, ValueError):
            if attempt == 3:
                return item["id"], "error"
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    selected = {"train": rows("train", args.train_count), "val": rows("val", args.val_count)}
    if set(x["id"] for x in selected["train"]) & set(x["id"] for x in selected["val"]):
        raise ValueError("Train/validation overlap")
    for split, items in selected.items():
        with (ROOT / f"selected_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item) + "\n")
    unique = {x["image_path"]: x for items in selected.values() for x in items}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, item) for item in unique.values()]
        for i, future in enumerate(as_completed(futures), 1):
            id_, status = future.result()
            if status == "error":
                failures.append(id_)
            if i % 100 == 0 or status == "error":
                print(i, "/", len(unique), "images", "failures", len(failures), flush=True)
    summary = {"train": len(selected["train"]), "val": len(selected["val"]),
               "unique_images": len(unique), "image_failures": failures}
    (ROOT / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)
    if failures:
        raise RuntimeError(f"Missing {len(failures)} images")


if __name__ == "__main__":
    main()
