"""Download the four unique model IDs used by both pipelines."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "facebook/dinov2-giant",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)


for model_id in MODELS:
    print(f"START {model_id}", flush=True)
    try:
        kwargs = {"max_workers": 4}
        if model_id == "meta-llama/Llama-3.1-8B-Instruct":
            kwargs["local_dir"] = Path(__file__).resolve().parent / "models" / "llama-3.1-8b-instruct"
        path = snapshot_download(model_id, **kwargs)
    except Exception as exc:
        print(f"ERROR {model_id}: {type(exc).__name__}: {exc}", flush=True)
        continue
    print(f"DONE {model_id}: {path}", flush=True)
