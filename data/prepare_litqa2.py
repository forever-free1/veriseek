import argparse
from pathlib import Path
from typing import Any

from veriseek_common import add_common_args, limit_rows, make_training_row, output_name, write_rows


def _get_first(example: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in example and example[key] not in (None, ""):
            return example[key]
    return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def is_litqa2(example: dict[str, Any]) -> bool:
    text = " ".join(str(example.get(key, "")) for key in ["subtask", "task", "dataset", "category"]).lower()
    return "litqa2" in text or "litqa" in text


def convert_example(example: dict[str, Any], split: str) -> dict[str, Any]:
    question = _get_first(example, ["question", "prompt", "input"])
    answer = _get_first(example, ["ideal", "answer", "target", "correct_answer"])
    distractors = _as_list(_get_first(example, ["distractors", "options", "choices"], []))
    evidence = _as_list(_get_first(example, ["sources", "evidence", "references"], []))
    options_text = "\n".join(f"- {item}" for item in [answer, *distractors] if str(item).strip())
    prompt = f"""You are a scientific research agent.

Answer the multiple-choice scientific question and include concise evidence.
Return your response using the exact format:

<answer>
selected answer
</answer>

<evidence>
[1] evidence sentence
</evidence>

Question:
{question}

Options:
{options_text}
"""
    return make_training_row(
        prompt=prompt,
        answer=answer,
        evidence=evidence,
        data_source="qasper_evidence",
        index=str(_get_first(example, ["id", "uid", "question_id"], "")),
        split=split,
        extra_info={"eval_only": True},
    )


def load_litqa2(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to download LAB-Bench/LitQA2: pip install datasets") from exc
    return load_dataset(dataset_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LitQA2 evaluation rows for VeriSeek.")
    add_common_args(parser)
    parser.add_argument("--dataset_name", default="futurehouse/lab-bench")
    args = parser.parse_args()

    dataset = load_litqa2(args.dataset_name)
    output_dir = Path(args.output_dir)
    for out_split, max_rows in {"train": args.max_train, "dev": args.max_dev, "test": args.max_test}.items():
        source_split = out_split if out_split in dataset else "train"
        rows_iter = (convert_example(item, out_split) for item in dataset[source_split] if is_litqa2(item))
        rows = limit_rows(rows_iter, max_rows)
        if rows:
            write_rows(rows, output_dir / output_name(out_split, args.write_jsonl), args.write_jsonl)
            print(f"wrote {len(rows)} {out_split} rows")


if __name__ == "__main__":
    main()
