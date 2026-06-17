import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metrics import (
    _data_source,
    _gold_answer_and_evidence,
    _ground_truth,
    _normalize_scifact_label,
    _prediction,
    _relaxed_answer,
    compute_components,
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    component_values: dict[str, list[float]] = defaultdict(list)
    gold_labels = Counter()
    pred_labels = Counter()
    confusion = Counter()

    for record in records:
        prediction = _prediction(record)
        ground_truth = _ground_truth(record)
        data_source = _data_source(record, task)
        components = compute_components(prediction, ground_truth, data_source)
        for key, value in components.items():
            component_values[key].append(float(value))

        gold_answer, _ = _gold_answer_and_evidence(ground_truth)
        pred_answer = _relaxed_answer(prediction, task) or "NO_LABEL"
        if task == "scifact":
            gold_answer = _normalize_scifact_label(gold_answer)
            pred_answer = _normalize_scifact_label(pred_answer)
        gold_labels[gold_answer] += 1
        pred_labels[pred_answer] += 1
        confusion[f"{gold_answer}->{pred_answer}"] += 1

    return {
        "count": len(records),
        "components": {key: _mean(values) for key, values in sorted(component_values.items())},
        "gold_label_distribution": dict(gold_labels),
        "pred_label_distribution": dict(pred_labels),
        "confusion_top": dict(confusion.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VeriSeek reward components for a prediction JSONL.")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--task", default="scifact", choices=["scifact", "qasper", "litqa2"])
    args = parser.parse_args()

    with Path(args.pred_path).open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(json.dumps(analyze(records, args.task), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
