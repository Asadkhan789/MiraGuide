"""Download local ScienceQA Parquet splits and dataset card."""

from pathlib import Path

from huggingface_hub import snapshot_download


root = Path(__file__).resolve().parent / "data" / "scienceqa"
path = snapshot_download(
    "derek-thomas/ScienceQA",
    repo_type="dataset",
    local_dir=root,
    allow_patterns=["data/*.parquet", "README.md"],
    max_workers=4,
)
print(path)
