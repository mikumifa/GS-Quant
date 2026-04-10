import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional


def best_match(
    targets: List[str],
    predicted: list[str],
    rank: int,
    topk_names: list[str],
) -> Optional[int]:
    pred = predicted[0]
    target = targets[0]
    if pred == target:
        return 1
    if pred not in set(topk_names):
        return rank + 1
    same_index = topk_names.index(pred)
    if same_index >= rank:
        return rank + 1
    return rank


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_label(s: str) -> str:
    s = s.lower()
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def normalize_prompt(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines)


def detect_direction(prompt: str) -> str:
    if "Here is a triplet with tail entity" in prompt:
        return "tail"
    if "Here is a triplet with head entity" in prompt:
        return "head"
    raise ValueError("Unable to detect prediction direction from prompt.")


def compute_metrics(
    ranks: Iterable[Optional[int]], hits_ks: List[int]
) -> Dict[str, float]:
    ranks_list = list(ranks)
    total = len(ranks_list)
    if total == 0:
        return {"count": 0, "mrr": 0.0, **{f"hits@{k}": 0.0 for k in hits_ks}}

    reciprocal_sum = 0.0
    hits_accumulators: Dict[int, int] = defaultdict(int)

    for rank in ranks_list:
        if rank is None:
            continue
        reciprocal_sum += 1.0 / rank
        for k in hits_ks:
            if rank <= k:
                hits_accumulators[k] += 1

    metrics = {"count": float(total), "mrr": reciprocal_sum / total}
    for k in hits_ks:
        metrics[f"hits@{k}"] = hits_accumulators[k] / total
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate KGC predictions using MRR and Hits@K, supporting multiple correct answers."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--hits", type=int, nargs="+", default=[1, 3, 10])

    parser.add_argument("--csv-out", type=Path, default=Path("rank_predictions.csv"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    references = read_jsonl(args.reference)
    ranks_by_direction: Dict[str, List[Optional[int]]] = {
        "head": [],
        "tail": [],
        "overall": [],
    }

    csv_rows = []

    for pred_record, reference in zip(predictions, references):
        prompt = pred_record["prompt"]
        direction = detect_direction(prompt)
        target_labels = [
            normalize_label(line)
            for line in reference["output"].splitlines()
            if line.strip()
        ]
        assert (
            prompt.strip() == f"{reference['instruction']}{reference['input']}".strip()
        )
        rank = reference["rank"]  # type:ignore
        topk_names = reference["topk_names"]  # type:ignore
        topk_names = [normalize_label(t) for t in topk_names]
        predicted_labels = [
            normalize_label(name)
            for name in pred_record["output"].splitlines()
            if name.strip()
        ]

        rank = best_match(
            target_labels,
            predicted_labels,
            rank,
            topk_names,
        )

        ranks_by_direction[direction].append(rank)
        ranks_by_direction["overall"].append(rank)
        csv_rows.append(
            {
                "direction": direction,
                "rank": rank if rank is not None else "None",
                "predicted_labels": " | ".join(predicted_labels),
                "target_labels": " | ".join(target_labels),
            }
        )

    for direction in ("tail", "head", "overall"):
        metrics = compute_metrics(ranks_by_direction[direction], args.hits)
        print(f"{direction.upper()}:")
        print(f"  Samples: {int(metrics['count'])}")
        print(f"  MRR: {metrics['mrr']:.4f}")
        for k in args.hits:
            print(f"  Hits@{k}: {metrics[f'hits@{k}']:.4f}")
        print()

    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "direction",
                "rank",
                "predicted_labels",
                "target_labels",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"\n✅ Saved detailed results with multi-answer ranking to: {args.csv_out.resolve()}"
    )


if __name__ == "__main__":
    main()
