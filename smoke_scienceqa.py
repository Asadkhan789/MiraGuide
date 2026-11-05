"""Run a Qwen2.5-VL ScienceQA baseline with per-question artifacts."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scienceqa_loader import DEFAULT_ROOT, load_scienceqa


ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def choose_rows(limit: int) -> list[dict]:
    image_target = limit // 2
    text_target = limit - image_target
    images, texts = [], []
    for sample in load_scienceqa("test", decode_images=True, include_references=True):
        if sample["image"] is not None and len(images) < image_target:
            images.append(sample)
        elif sample["image"] is None and len(texts) < text_target:
            texts.append(sample)
        if len(images) == image_target and len(texts) == text_target:
            break
    if len(images) != image_target or len(texts) != text_target:
        raise ValueError(f"Could select only {len(images)} image and {len(texts)} text rows")
    return sorted(images + texts, key=lambda row: int(row["id"].split("-")[-1]))


def build_messages(sample: dict) -> list[dict]:
    choices = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample["choices"]))
    prompt = (
        "Answer this multiple-choice question using the question, choices, hint, and image if provided. "
        "Give a brief explanation, then finish with exactly one line: Answer: <choice text>.\n\n"
        f"Question: {sample['question']}\n"
        f"Choices:\n{choices}\n"
        f"Hint: {sample['hint'] or '(none)'}"
    )
    content = ([{"type": "image"}] if sample["image"] is not None else []) + [{"type": "text", "text": prompt}]
    return [{"role": "user", "content": content}]


def extract_answer(raw: str, choices: list[str]) -> str:
    match = re.search(r"(?im)^\s*Answer:\s*(.+?)\s*$", raw)
    answer = match.group(1).strip() if match else raw.strip().splitlines()[-1].strip()
    answer = re.sub(r"^[A-Z][.)]\s*", "", answer)
    if len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = ord(answer.upper()) - 65
        if idx < len(choices):
            return choices[idx]
    for choice in choices:
        if normalize(answer) == normalize(choice):
            return choice
    return answer


def extract_explanation(raw: str) -> str:
    return re.split(r"(?im)^\s*Answer:\s*", raw, maxsplit=1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "scienceqa_baseline")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = choose_rows(args.limit)
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    ).eval()
    manifest = []

    for number, sample in enumerate(rows, start=1):
        folder = args.output / f"q{number:02d}"
        folder.mkdir(exist_ok=True)
        image = sample["image"]
        input_record = {k: v for k, v in sample.items() if k not in {"image", "answer", "answer_index", "lecture", "solution"}}
        input_record["has_image"] = image is not None
        write_json(folder / "00_input.json", input_record)
        if image is not None:
            image.convert("RGB").save(folder / "00_image.png")
        messages = build_messages(sample)
        write_json(folder / "01_prompt.json", messages)
        result = {"number": number, "id": sample["id"], "has_image": image is not None, "status": "started"}
        manifest.append(result)
        write_json(args.output / "manifest.json", manifest)
        try:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[text], images=[image] if image is not None else None,
                               return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            answer = extract_answer(raw, sample["choices"])
            write_json(folder / "02_prediction.json", {"raw_text": raw,
                                                     "explanation": extract_explanation(raw),
                                                     "answer": answer})
            correct = normalize(answer) == normalize(sample["answer"])
            write_json(folder / "03_reference_check.json", {
                "answer_index": sample["answer_index"], "reference_answer": sample["answer"],
                "reference_solution": sample["solution"], "correct": correct,
            })
            result.update(status="complete", answer=answer, correct=correct)
        except Exception as exc:
            result["status"] = f"error: {type(exc).__name__}: {exc}"
        write_json(args.output / "manifest.json", manifest)
        print(f"q{number:02d} {result['status']}", flush=True)

    complete = [x for x in manifest if x["status"] == "complete"]
    write_json(args.output / "summary.json", {
        "model": MODEL_ID, "requested": args.limit, "completed": len(complete),
        "correct": sum(x["correct"] for x in complete),
        "accuracy_on_completed": (sum(x["correct"] for x in complete) / len(complete)) if complete else None,
        "image_questions": sum(x["has_image"] for x in manifest),
        "text_questions": sum(not x["has_image"] for x in manifest),
    })


if __name__ == "__main__":
    main()
