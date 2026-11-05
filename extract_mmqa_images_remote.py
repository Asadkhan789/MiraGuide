"""Fetch images referenced by prepared MMQA rows using HTTP byte ranges."""

import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from download_mmqa import IMAGE_URL


ROOT = Path(__file__).resolve().parent / "data" / "mmqa"


def range_bytes(start, end):
    for attempt in range(4):
        try:
            response = requests.get(IMAGE_URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 90))
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise IOError(f"Bad range response: {response.status_code}")
            return response.content
        except (requests.RequestException, IOError):
            if attempt == 3:
                raise


def index_zip():
    head = requests.head(IMAGE_URL, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    tail = range_bytes(size - 65536, size - 1)
    eocd = tail.rfind(bytes((80, 75, 5, 6)))
    if eocd < 0:
        raise ValueError("ZIP end record missing")
    _, disk, cd_disk, _, count, cd_size, cd_offset, _ = struct.unpack_from("<4s4H2LH", tail, eocd)
    if disk or cd_disk:
        raise ValueError("Multi-disk ZIP unsupported")
    central = range_bytes(cd_offset, cd_offset + cd_size - 1)
    entries = {}
    pos = 0
    for _ in range(count):
        if central[pos:pos+4] != bytes((80, 75, 1, 2)):
            raise ValueError(f"Bad central directory entry at {pos}")
        h = struct.unpack_from("<4s6H3L5H2L", central, pos)
        method, crc, compressed, uncompressed = h[4], h[7], h[8], h[9]
        name_len, extra_len, comment_len, local_offset = h[10], h[11], h[12], h[16]
        name = central[pos+46:pos+46+name_len].decode("utf-8", errors="replace")
        basename = Path(name).name
        if basename:
            if basename in entries:
                raise ValueError(f"Duplicate image basename: {basename}")
            entries[basename] = (local_offset, compressed, uncompressed, method, crc)
        pos += 46 + name_len + extra_len + comment_len
    if pos != len(central):
        raise ValueError("Central directory length mismatch")
    return entries


def fetch(name, meta):
    target = ROOT / "images" / name
    if target.exists() and target.stat().st_size == meta[2]:
        return
    offset, compressed, uncompressed, method, crc = meta
    header = range_bytes(offset, offset + 29)
    if header[:4] != bytes((80, 75, 3, 4)):
        raise ValueError(f"Bad local ZIP header for {name}")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    start = offset + 30 + name_len + extra_len
    data = range_bytes(start, start + compressed - 1)
    if method == 8:
        data = zlib.decompress(data, -15)
    elif method != 0:
        raise ValueError(f"Unsupported ZIP compression {method}: {name}")
    if len(data) != uncompressed or zlib.crc32(data) != crc:
        raise ValueError(f"Image checksum mismatch: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    part.write_bytes(data)
    part.replace(target)


def main():
    wanted = set()
    for split in ("train", "val"):
        for line in (ROOT / f"selected_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            name = json.loads(line).get("image_name")
            if name:
                wanted.add(name)
    entries = index_zip()
    missing = wanted - entries.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced MMQA images missing from archive")
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, name, entries[name]) for name in sorted(wanted)]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 25 == 0:
                print(f"images {i}/{len(wanted)}", flush=True)
    print(f"Extracted {len(wanted)} referenced MMQA images", flush=True)


if __name__ == "__main__":
    main()
