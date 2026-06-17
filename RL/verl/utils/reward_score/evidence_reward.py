import json
import re
import string
from collections import Counter
from typing import Any


LABEL_ALIASES = {
    "SUPPORT": "SUPPORTS",
    "SUPPORTS": "SUPPORTS",
    "SUPPORTED": "SUPPORTS",
    "REFUTE": "REFUTES",
    "REFUTES": "REFUTES",
    "REFUTED": "REFUTES",
    "CONTRADICT": "REFUTES",
    "CONTRADICTS": "REFUTES",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
    "NEI": "NOT_ENOUGH_INFO",
}


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text or "", flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_evidence(text: str) -> list[str]:
    match = re.search(r"<evidence>(.*?)</evidence>", text or "", flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    body = match.group(1).strip()
    if not body:
        return []
    items = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\[\d+\]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if line:
            items.append(line)
    return items or [body]


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text).lower()
    text = text.replace("_", " ")
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def format_reward(solution_str: str) -> float:
    has_answer = bool(re.search(r"<answer>.*?</answer>", solution_str or "", flags=re.IGNORECASE | re.DOTALL))
    has_evidence = bool(re.search(r"<evidence>.*?</evidence>", solution_str or "", flags=re.IGNORECASE | re.DOTALL))
    if has_answer and has_evidence:
        return 1.0
    if has_answer or has_evidence:
        return 0.5
    return 0.0


def conciseness_reward(evidence_items: list[str]) -> float:
    return 1.0 if 1 <= len(evidence_items) <= 5 else 0.0


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        data = ground_truth
    elif isinstance(ground_truth, str):
        try:
            data = json.loads(ground_truth)
        except json.JSONDecodeError:
            data = {"answer": ground_truth, "evidence": []}
    else:
        data = {"answer": str(ground_truth), "evidence": []}

    answer = data.get("answer", data.get("label", data.get("target", "")))
    evidence = data.get("evidence", data.get("evidences", data.get("gold_evidence", [])))
    if isinstance(evidence, str):
        evidence = [evidence]
    elif evidence is None:
        evidence = []
    return {"answer": answer, "evidence": [str(item) for item in evidence if str(item).strip()]}


def _normalize_label(label: Any) -> str:
    normalized = normalize_text(label).upper().replace(" ", "_")
    return LABEL_ALIASES.get(normalized, normalized)


def _answer_reward(pred_answer: str, gold_answer: Any, data_source: str) -> float:
    if data_source == "scifact_evidence":
        return 1.0 if _normalize_label(pred_answer) == _normalize_label(gold_answer) else 0.0
    return token_f1(pred_answer, gold_answer)


def _evidence_reward(pred_evidence: list[str], gold_evidence: list[str]) -> float:
    if not pred_evidence or not gold_evidence:
        return 0.0
    scores = [max(token_f1(pred, gold) for gold in gold_evidence) for pred in pred_evidence]
    return sum(scores) / len(scores)


def compute_components(solution_str: str, ground_truth: Any, data_source: str) -> dict[str, float]:
    parsed_gold = _parse_ground_truth(ground_truth)
    pred_answer = extract_answer(solution_str)
    pred_evidence = extract_evidence(solution_str)
    return {
        "answer": _answer_reward(pred_answer, parsed_gold["answer"], data_source),
        "evidence": _evidence_reward(pred_evidence, parsed_gold["evidence"]),
        "format": format_reward(solution_str),
        "conciseness": conciseness_reward(pred_evidence),
    }


def _empty_or_concise_nei_reward(pred_evidence: list[str]) -> float:
    if not pred_evidence:
        return 1.0
    if len(pred_evidence) <= 2 and all(len(normalize_text(item).split()) <= 20 for item in pred_evidence):
        return 0.5
    return 0.0


def _scifact_gated_score(solution_str: str, ground_truth: Any) -> float:
    parsed_gold = _parse_ground_truth(ground_truth)
    pred_answer = extract_answer(solution_str)
    pred_evidence = extract_evidence(solution_str)
    components = compute_components(solution_str, ground_truth, "scifact_evidence")

    if components["format"] < 1.0:
        return 0.0

    gold_label = _normalize_label(parsed_gold["answer"])
    pred_label = _normalize_label(pred_answer)
    if gold_label == "NOT_ENOUGH_INFO":
        return 0.8 * components["answer"] + 0.2 * _empty_or_concise_nei_reward(pred_evidence)

    if pred_label == "NOT_ENOUGH_INFO":
        return 0.05

    score = 0.35 * components["answer"] + 0.55 * components["evidence"] + 0.10 * components["conciseness"]
    if components["evidence"] < 0.2:
        score = min(score, 0.25)
    return score


def compute_score(solution_str: str, ground_truth: Any, data_source: str = "qasper_evidence", val_type: str = "f1") -> float:
    components = compute_components(solution_str, ground_truth, data_source)
    if data_source == "scifact_evidence":
        return _scifact_gated_score(solution_str, ground_truth)
    return (
        0.45 * components["answer"]
        + 0.35 * components["evidence"]
        + 0.15 * components["format"]
        + 0.05 * components["conciseness"]
    )
