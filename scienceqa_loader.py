"""Load local ScienceQA parquet splits without leaking labels into model inputs."""

from pathlib import Path
from typing import Iterator

from datasets import Image, load_dataset


DEFAULT_ROOT = Path(__file__).resolve().parent / "data" / "scienceqa"
SPLITS = ("train", "validation", "test")


def load_scienceqa(
    split: str = "test",
    root: str | Path = DEFAULT_ROOT,
    limit: int | None = None,
    decode_images: bool = False,
    include_references: bool = False,
) -> Iterator[dict]:
    """Yield normalized ScienceQA examples from the downloaded Parquet files.

    `include_references=False` keeps the gold answer and solution out of inputs.
    Images are left encoded by default; set `decode_images=True` for PIL images.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")

    root = Path(root)
    files = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No {split} Parquet files under {root / 'data'}. "
            "Download derek-thomas/ScienceQA into this directory first."
        )

    dataset = load_dataset("parquet", data_files={split: [str(p) for p in files]}, split=split)
    if "image" in dataset.column_names:
        dataset = dataset.cast_column("image", Image(decode=decode_images))

    count = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(count):
        row = dataset[index]
        choices = list(row.get("choices") or [])
        sample = {
            "id": f"scienceqa-{split}-{index}",
            "split": split,
            "question": row["question"],
            "choices": choices,
            "hint": row.get("hint") or "",
            "image": row.get("image"),
            "task": row.get("task") or "",
            "grade": row.get("grade") or "",
            "subject": row.get("subject") or "",
            "topic": row.get("topic") or "",
            "category": row.get("category") or "",
            "skill": row.get("skill") or "",
        }
        if include_references:
            answer_index = int(row["answer"])
            sample["answer_index"] = answer_index
            sample["answer"] = choices[answer_index]
            sample["lecture"] = row.get("lecture") or ""
            sample["solution"] = row.get("solution") or ""
        yield sample


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview locally downloaded ScienceQA")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    for sample in load_scienceqa(args.split, args.root, args.limit):
        print(sample["id"], sample["question"], sample["choices"])
