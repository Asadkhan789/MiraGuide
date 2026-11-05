"""Extract images referenced by the prepared MMQA train/dev rows."""

import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data" / "mmqa"


def main():
    wanted = set()
    for split in ("train", "val"):
        for line in (ROOT / f"selected_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("image_name"):
                wanted.add(row["image_name"])
    archive = ROOT / "final_dataset_images.zip"
    with zipfile.ZipFile(archive) as zipped:
        matches = {}
        for name in zipped.namelist():
            basename = Path(name).name
            if basename in wanted:
                if basename in matches:
                    raise ValueError(f"Duplicate archive image: {basename}")
                matches[basename] = name
        missing = wanted - matches.keys()
        if missing:
            raise FileNotFoundError(f"{len(missing)} referenced images absent from ZIP, e.g. {next(iter(missing))}")
        output = ROOT / "images"
        output.mkdir(parents=True, exist_ok=True)
        for i, basename in enumerate(sorted(wanted), 1):
            target = output / basename
            if not target.exists():
                with zipped.open(matches[basename]) as source, target.open("wb") as dest:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
            if i % 50 == 0:
                print(f"images {i}/{len(wanted)}", flush=True)
    print(f"Extracted {len(wanted)} referenced MMQA images", flush=True)


if __name__ == "__main__":
    main()
