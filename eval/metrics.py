import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = ROOT / "RL" / "verl" / "utils" / "reward_score" / "evidence_reward.py"
_SPEC = importlib.util.spec_from_file_location("evidence_reward", REWARD_PATH)
_REWARD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REWARD)

compute_components = _REWARD.compute_components
extract_answer = _REWARD.extract_answer
extract_evidence = _REWARD.extract_evidence
format_reward = _REWARD.format_reward
token_f1 = _REWARD.token_f1
normalize_text = _REWARD.normalize_text


SCIFACT_RELAXED_PATTERNS = [
    (
        "NOT_ENOUGH_INFO",
        [
            r"\bnot enough info(?:rmation)?\b",
            r"\bnei\b",
            r"\binsufficient evidence\b",
            r"\bnot enough evidence\b",
            r"\bcannot be determined\b",
            r"\bcan't be determined\b",
            r"\bunknown\b",
        ],
    ),
    (
        "REFUTES",
        [
            r"\brefutes?\b",
            r"\brefuted\b",
            r"\bcontradicts?\b",
            r"\bcontradicted\b",
            r"\bnot supported\b",
            r"\bfalse\b",
        ],
    ),
    (
        "SUPPORTS",
        [
            r"\bsupports?\b",
            r"\bsupported\b",
            r"\bconsistent with\b",
            r"\btrue\b",
        ],
    ),
]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _prediction(record: dict[str, Any]) -> str:
    for key in ["prediction", "response", "output", "solution", "text"]:
        if key in record:
            value = record[key]
            if isinstance(value, list):
                return str(value[0]) if value else ""
            return str(value)
    return ""


def _ground_truth(record: dict[str, Any]) -> Any:
    if "ground_truth" in record:
        return record["ground_truth"]
    reward_model = record.get("reward_model") or {}
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth", {})
    return {}


def _data_source(record: dict[str, Any], task: str) -> str:
    if record.get("data_source"):
        return str(record["data_source"])
    return {
        "scifact": "scifact_evidence",
        "qasper": "qasper_evidence",
        "litqa2": "litqa2_evidence",
    }[task]


def _gold_answer_and_evidence(ground_truth: Any) -> tuple[str, list[str]]:
    if isinstance(ground_truth, str):
        try:
            data = json.loads(ground_truth)
        except json.JSONDecodeError:
            data = {"answer": ground_truth, "evidence": []}
    elif isinstance(ground_truth, dict):
        data = ground_truth
    else:
        data = {"answer": str(ground_truth), "evidence": []}
    evidence = data.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    return str(data.get("answer", "")), [str(item) for item in evidence]


def _normalize_scifact_label(label: Any) -> str:
    normalized = normalize_text(label).upper().replace(" ", "_")
    aliases = getattr(_REWARD, "LABEL_ALIASES", {})
    return aliases.get(normalized, normalized)


def _relaxed_scifact_answer(prediction: str) -> str:
    strict_answer = extract_answer(prediction)
    if strict_answer:
        return strict_answer
    text = prediction or ""
    for label, patterns in SCIFACT_RELAXED_PATTERNS:
        for pattern in patterns:
            if re_search(pattern, text):
                return label
    return ""


def re_search(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text or "", flags=re.IGNORECASE))


def _relaxed_answer(prediction: str, task: str) -> str:
    if task == "scifact":
        return _relaxed_scifact_answer(prediction)
    strict_answer = extract_answer(prediction)
    return strict_answer if strict_answer else prediction


def _relaxed_evidence_text(prediction: str) -> str:
    strict_evidence = extract_evidence(prediction)
    if strict_evidence:
        return " ".join(strict_evidence)
    return prediction or ""


def _relaxed_answer_score(prediction: str, gold_answer: str, task: str) -> float:
    pred_answer = _relaxed_answer(prediction, task)
    if task == "scifact":
        return 1.0 if _normalize_scifact_label(pred_answer) == _normalize_scifact_label(gold_answer) else 0.0
    return token_f1(pred_answer, gold_answer)


def unsupported_answer_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    unsupported = 0
    for record in records:
        pred = _prediction(record)
        if extract_answer(pred) and not extract_evidence(pred):
            unsupported += 1
    return unsupported / len(records)


def evaluate_records(records: list[dict[str, Any]], task: str, mode: str = "both") -> dict[str, float]:
    if mode not in {"strict", "relaxed", "both"}:
        raise ValueError("mode must be one of: strict, relaxed, both")
    if not records:
        return {"count": 0}
    answer_scores = []
    evidence_scores = []
    format_scores = []
    exact_scores = []
    evidence_presence = []
    relaxed_answer_scores = []
    relaxed_evidence_scores = []
    for record in records:
        pred = _prediction(record)
        gt = _ground_truth(record)
        data_source = _data_source(record, task)
        components = compute_components(pred, gt, data_source)
        answer_scores.append(components["answer"])
        evidence_scores.append(components["evidence"])
        format_scores.append(1.0 if format_reward(pred) == 1.0 else 0.0)
        evidence_presence.append(1.0 if extract_evidence(pred) else 0.0)
        gold_answer, _ = _gold_answer_and_evidence(gt)
        exact_scores.append(1.0 if token_f1(extract_answer(pred), gold_answer) == 1.0 else 0.0)
        gold_answer, gold_evidence = _gold_answer_and_evidence(gt)
        relaxed_answer_scores.append(_relaxed_answer_score(pred, gold_answer, task))
        relaxed_evidence_scores.append(token_f1(_relaxed_evidence_text(pred), " ".join(gold_evidence)))

    result = {"count": len(records)}
    if mode in {"strict", "both"}:
        result.update(
            {
                "evidence_f1": sum(evidence_scores) / len(evidence_scores),
                "format_success_rate": sum(format_scores) / len(format_scores),
                "unsupported_answer_rate": unsupported_answer_rate(records),
            }
        )
        if task == "scifact":
            result["label_accuracy"] = sum(answer_scores) / len(answer_scores)
        elif task == "qasper":
            result["answer_f1"] = sum(answer_scores) / len(answer_scores)
        elif task == "litqa2":
            result["multiple_choice_accuracy"] = sum(exact_scores) / len(exact_scores)
            result["evidence_block_presence"] = sum(evidence_presence) / len(evidence_presence)

    if mode in {"relaxed", "both"}:
        if task == "scifact":
            result["relaxed_label_accuracy"] = sum(relaxed_answer_scores) / len(relaxed_answer_scores)
        elif task == "qasper":
            result["relaxed_answer_f1"] = sum(relaxed_answer_scores) / len(relaxed_answer_scores)
        elif task == "litqa2":
            result["relaxed_multiple_choice_accuracy"] = sum(relaxed_answer_scores) / len(relaxed_answer_scores)
        result["relaxed_full_text_evidence_f1"] = sum(relaxed_evidence_scores) / len(relaxed_evidence_scores)
    return result


def run_cli(task: str) -> None:
    parser = argparse.ArgumentParser(description=f"Evaluate VeriSeek predictions for {task}.")
    parser.add_argument("--pred_path", required=True, help="JSONL file with prediction/ground_truth records.")
    parser.add_argument("--gold_path", default=None, help="Optional JSONL gold file; unused when predictions include ground_truth.")
    parser.add_argument(
        "--mode",
        choices=["strict", "relaxed", "both"],
        default="both",
        help="Evaluation mode. strict preserves protocol-only metrics; relaxed scans natural-language outputs.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.pred_path)
    if args.gold_path:
        gold_records = load_jsonl(args.gold_path)
        for pred, gold in zip(records, gold_records):
            pred.setdefault("ground_truth", _ground_truth(gold))
            pred.setdefault("data_source", gold.get("data_source"))
    print(json.dumps(evaluate_records(records, task, mode=args.mode), indent=2, ensure_ascii=False))
