"""ROUGE-L and BERTScore F1 against nonempty ScienceQA solutions."""

import argparse
import json
from pathlib import Path

from bert_score import score as bert_score
from rouge_score.rouge_scorer import RougeScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    examples = json.loads((args.run / "generated_explanations.json").read_text(encoding="utf-8"))
    usable = [x for x in examples if x.get("reference_solution", "").strip()]
    if not usable:
        raise ValueError("No nonempty ScienceQA reference solutions")
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    references = [x["reference_solution"] for x in usable]
    result = {"n_generated": len(examples), "n_scored": len(usable),
              "reference": "nonempty ScienceQA solution",
              "rouge_l": "Google rouge-score, F1, use_stemmer=True",
              "bert_score": "allenai/scibert_scivocab_uncased, layer 9, F1, no IDF, no baseline rescaling",
              "arms": {}}
    for arm in ("off", "on"):
        candidates = [x[f"{arm}_explanation"] for x in usable]
        rouge = [scorer.score(ref, cand)["rougeL"].fmeasure
                 for ref, cand in zip(references, candidates)]
        _, _, f1 = bert_score(candidates, references, model_type="allenai/scibert_scivocab_uncased",
                              num_layers=9, idf=False, rescale_with_baseline=False,
                              batch_size=16, device="cuda", verbose=False)
        bert = f1.cpu().tolist()
        result["arms"][arm] = {
            "rouge_l_f1_mean": sum(rouge) / len(rouge),
            "bertscore_f1_mean": sum(bert) / len(bert),
            "rows": [{"id": row["id"], "rouge_l_f1": r, "bertscore_f1": b}
                     for row, r, b in zip(usable, rouge, bert)]}
    output = args.run / "explanation_rouge_bert.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"n_generated": result["n_generated"], "n_scored": result["n_scored"],
                      "arms": {k: {m: v for m, v in arm.items() if m != "rows"}
                               for k, arm in result["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
