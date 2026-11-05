"""Score VQA-X validation explanations against their reference explanations."""

import argparse
import json
from pathlib import Path

from bert_score import score as bert_score
from rouge_score.rouge_scorer import RougeScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / "validation_predictions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if not rows or any(not r["reference_explanations"] for r in rows):
        raise ValueError("Need labeled VQA-X validation rows with reference explanations")
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    references = [r["reference_explanations"][0] for r in rows]
    result = {"count": len(rows), "reference": "first VQA-X reference explanation",
              "rouge_l": "F1, stemming", "bert_score": "SciBERT layer 9 F1", "arms": {}}
    for arm in ("off", "on"):
        candidates = [r[f"{arm}_explanation"] for r in rows]
        rouge = [scorer.score(ref, cand)["rougeL"].fmeasure
                 for ref, cand in zip(references, candidates)]
        _, _, f1 = bert_score(candidates, references,
                              model_type="allenai/scibert_scivocab_uncased", num_layers=9,
                              idf=False, rescale_with_baseline=False,
                              batch_size=16, device="cuda", verbose=False)
        result["arms"][arm] = {"rouge_l_f1_mean": sum(rouge) / len(rouge),
                               "bertscore_f1_mean": float(f1.mean()),
                               "rows": [{"id": row["id"], "rouge_l_f1": r,
                                         "bertscore_f1": float(b)}
                                        for row, r, b in zip(rows, rouge, f1)]}
    (args.run / "explanation_rouge_bert.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({arm: {k: v for k, v in score.items() if k != "rows"}
                      for arm, score in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
