"""Train a ScienceQA fusion connector with a frozen Llama decoder.

Train examples come from ScienceQA train. Evaluation uses held-out ScienceQA test
examples with the same prompt in both arms. A learned image/text prefix is the
only difference between arms. This is an ablation, not full MIRA training.
"""

import argparse
import io
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer

from scienceqa_loader import load_scienceqa
from smoke_scienceqa import choose_rows


ROOT = Path(__file__).resolve().parent
STAGED = ROOT / "outputs" / "scienceqa_stages"
MODEL = Path(os.environ.get("MIRA_LLAMA_MODEL", ROOT / "models" / "llama-3.1-8b-instruct"))


class Connector(nn.Module):
    """Trainable fusion projection and a two-source router producing one token."""

    def __init__(self, hidden: int, visual: int = 1536):
        super().__init__()
        self.text_proj = nn.Linear(hidden, hidden, bias=False)
        self.image_proj = nn.Linear(visual, hidden, bias=False)
        self.router = nn.Sequential(nn.Linear(hidden * 2, 64), nn.Tanh(), nn.Linear(64, 2))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.text_proj(text.float())
        v = self.image_proj(image.float())
        weights = torch.softmax(self.router(torch.cat((t, v), dim=-1)), dim=-1)
        fused = weights[0] * t + weights[1] * v
        return self.norm(fused).unsqueeze(0), weights


def image_from_row(row: dict) -> Image.Image | None:
    payload = row.get("image")
    if not payload:
        return None
    if isinstance(payload, Image.Image):
        return payload.convert("RGB")
    if isinstance(payload, dict) and payload.get("bytes"):
        return Image.open(io.BytesIO(payload["bytes"])).convert("RGB")
    if isinstance(payload, dict) and payload.get("path"):
        return Image.open(payload["path"]).convert("RGB")
    return None


def build_prompt(row: dict, evidence: str = "") -> str:
    choices = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(row["choices"]))
    hint = row.get("hint") or ""
    return ("Choose the best answer. Reply with one choice letter only.\n"
            f"Question: {row['question']}\nChoices:\n{choices}\n"
            f"Hint: {hint}\nEvidence: {evidence}\nAnswer:")


def select_train(count: int | None) -> list[dict]:
    selected = []
    image_count = text_count = 0
    for row in load_scienceqa("train", include_references=True):
        has_image = row["image"] is not None
        if count is not None and (len(row["choices"]) > 5 or len(row["question"]) + len(row["hint"]) > 850):
            continue
        if count is None:
            selected.append(row)
        elif has_image and image_count < count // 2:
            selected.append(row)
            image_count += 1
        elif not has_image and text_count < count - count // 2:
            selected.append(row)
            text_count += 1
        if count is not None and len(selected) == count:
            break
    if count is not None and len(selected) != count:
        raise RuntimeError(f"Found only {len(selected)} suitable training examples")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "fusion_ablation")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch required; activate the pytorch conda environment")
    torch.manual_seed(17)
    args.output.mkdir(parents=True, exist_ok=True)
    train = select_train(args.train_count)
    test = choose_rows(args.test_count)
    if {r["id"] for r in train} & {r["id"] for r in test}:
        raise RuntimeError("Train/test overlap")
    split = {"train_ids": [r["id"] for r in train], "test_ids": [r["id"] for r in test]}
    (args.output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    device = torch.device("cuda")
    print("Precomputing frozen DINOv2 features", flush=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant", use_fast=True)
    vision = AutoModel.from_pretrained("facebook/dinov2-giant", torch_dtype=torch.float16).to(device).eval()
    visual_features = {}
    with torch.inference_mode():
        for row in train + test:
            image = image_from_row(row)
            if image is None:
                visual_features[row["id"]] = torch.zeros(1536)
                continue
            inputs = processor(images=image, return_tensors="pt").to(device)
            visual_features[row["id"]] = vision(**inputs).last_hidden_state[:, 0].float().cpu().squeeze(0)
    del vision, processor
    torch.cuda.empty_cache()

    print("Loading frozen Llama", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llama = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                  attn_implementation="sdpa").to(device).eval()
    llama.requires_grad_(False)
    llama.gradient_checkpointing_enable()
    connector = Connector(llama.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=2e-5)
    embedding = llama.get_input_embeddings()
    letter_ids = [tokenizer.encode(" " + chr(65+i), add_special_tokens=False)[-1] for i in range(5)]

    def forward(row: dict, use_connector: bool, evidence: str = "", train_mode: bool = False,
                blank_image: bool = False):
        prompt = build_prompt(row, evidence)
        token_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=420).input_ids.to(device)
        embeds = embedding(token_ids)
        weights = None
        if use_connector:
            text_vec = embeds.detach().float().mean(dim=1).squeeze(0)
            image_vec = visual_features[row["id"]].to(device)
            if blank_image:
                image_vec = torch.zeros_like(image_vec)
            prefix, weights = connector(text_vec, image_vec)
            embeds = torch.cat((prefix.to(embeds.dtype).unsqueeze(0), embeds), dim=1)
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
        logits = llama(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits[0, -1]
        candidate_logits = logits[letter_ids[:len(row["choices"])]]
        if train_mode:
            loss = nn.functional.cross_entropy(candidate_logits.float().unsqueeze(0),
                                               torch.tensor([row["answer_index"]], device=device))
            return loss, weights
        return int(candidate_logits.argmax()), weights

    history = []
    for epoch in range(args.epochs):
        connector.train()
        for row in train:
            optimizer.zero_grad(set_to_none=True)
            loss, weights = forward(row, True, train_mode=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
            optimizer.step()
            record = {"epoch": epoch + 1, "id": row["id"], "loss": float(loss.detach()),
                      "router": weights.detach().cpu().tolist()}
            history.append(record)
            print(json.dumps(record), flush=True)
    torch.save({"state_dict": connector.cpu().state_dict(), "train_ids": split["train_ids"]},
               args.output / "connector.pt")
    connector.to(device).eval()
    (args.output / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    results = []
    for index, row in enumerate(test, 1):
        folder = STAGED / f"q{index:02d}"
        input_file = folder / "00_input.json"
        if input_file.exists() and json.loads(input_file.read_text(encoding="utf-8"))["id"] != row["id"]:
            raise RuntimeError(f"Staged evidence mismatch for {row['id']}")
        staged = folder / "04_evidence_prompt.json"
        evidence = "\n".join(json.loads(staged.read_text(encoding="utf-8"))["evidence_clauses"]) if staged.exists() else ""
        caption_file = folder / "01_caption.json"
        if caption_file.exists():
            caption = json.loads(caption_file.read_text(encoding="utf-8")).get("caption", "")
            evidence = "\n".join(x for x in (caption, evidence) if x)
        with torch.no_grad():
            off, _ = forward(row, False, evidence)
            on, weights = forward(row, True, evidence)
            on_blank_image, _ = forward(row, True, evidence, blank_image=True)
        record = {"id": row["id"], "has_image": row["image"] is not None,
                  "gold": row["answer_index"], "off": off, "on": on,
                  "on_blank_image": on_blank_image,
                  "off_correct": off == row["answer_index"],
                  "on_correct": on == row["answer_index"],
                  "on_blank_image_correct": on_blank_image == row["answer_index"],
                  "router": weights.cpu().tolist()}
        results.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"train_count": len(train), "test_count": len(test),
               "off_correct": sum(x["off_correct"] for x in results),
               "on_correct": sum(x["on_correct"] for x in results),
               "on_blank_image_correct": sum(x["on_blank_image_correct"] for x in results),
               "note": "One learned prefix token; not full MIRA KGJSA attention training."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
