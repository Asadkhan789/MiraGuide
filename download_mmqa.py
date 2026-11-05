"""Download official AllenAI MultiModalQA train/dev contexts and image archive."""

import argparse
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent / "data" / "mmqa"
FILES = ("MMQA_train.jsonl.gz", "MMQA_dev.jsonl.gz", "MMQA_tables.jsonl.gz",
         "MMQA_texts.jsonl.gz", "MMQA_images.jsonl.gz")
BASE = "https://raw.githubusercontent.com/allenai/multimodalqa/master/dataset/"
IMAGE_URL = "https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if target.exists():
        print("exists", target, flush=True)
        return
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        with part.open("ab" if offset else "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    if handle.tell() % (100 * 1024 * 1024) < len(chunk):
                        print(target.name, round(handle.tell() / 1e6), "MB", flush=True)
    part.replace(target)
    print("done", target.name, target.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-image-archive", action="store_true",
                        help="Download annotations only; fetch referenced images with extract_mmqa_images_remote.py")
    args = parser.parse_args()
    for name in FILES:
        download(BASE + name, ROOT / name)
    if not args.skip_image_archive:
        download(IMAGE_URL, ROOT / "final_dataset_images.zip")


if __name__ == "__main__":
    main()
