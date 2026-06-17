import argparse
from pathlib import Path
from typing import Any

from veriseek_common import add_common_args, limit_rows, make_training_row, output_name, write_rows


def _first_nonempty(values: list[Any]) -> str:
    for value in values:
        if isinstance(value, list) and value:
            return "; ".join(str(item) for item in value if str(item).strip())
        if value is not None and str(value).strip():
            return str(value)
    return ""


def answer_text(answer: dict[str, Any]) -> str:
    if answer.get("unanswerable"):
        return "unanswerable"
    extractive = answer.get("extractive_spans") or []
    free_form = answer.get("free_form_answer") or ""
    if extractive:
        return "; ".join(str(item) for item in extractive)
    if free_form:
        return str(free_form)
    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "yes" if yes_no else "no"
    return ""


def answer_evidence(answer: dict[str, Any]) -> list[str]:
    evidence = answer.get("highlighted_evidence") or answer.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    return [str(item) for item in evidence if str(item).strip()]


def flatten_context(paper: dict[str, Any], max_chars: int = 12000) -> str:
    chunks = []
    abstract = paper.get("abstract", "")
    if abstract:
        chunks.append(f"Abstract:\n{abstract}")
    full_text = paper.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    paragraphs = full_text.get("paragraphs") or []
    for section, section_paragraphs in zip(section_names, paragraphs):
        section_text = "\n".join(str(p) for p in section_paragraphs)
        chunks.append(f"{section}:\n{section_text}")
    return "\n\n".join(chunks)[:max_chars]


def convert_paper(paper: dict[str, Any], split: str, max_context_chars: int = 12000) -> list[dict[str, Any]]:
    qas = paper.get("qas") or {}
    questions = qas.get("question") or []
    question_ids = qas.get("question_id") or [str(i) for i in range(len(questions))]
    answers_by_question = qas.get("answers") or []
    context = flatten_context(paper, max_context_chars)
    rows = []
    for idx, question in enumerate(questions):
        answer_group = answers_by_question[idx] if idx < len(answers_by_question) else {}
        answers = answer_group.get("answer", []) if isinstance(answer_group, dict) else []
        if not answers:
            continue
        answer = answers[0]
        gold_answer = answer_text(answer)
        if not gold_answer:
            continue
        prompt = f"""You are a scientific research agent.

Answer the question using explicit evidence from the paper context.
Return your response using the exact format:

<answer>
your answer
</answer>

<evidence>
[1] evidence sentence
</evidence>

Paper title:
{paper.get("title", "")}

Question:
{question}

Available paper context:
{context}
"""
        rows.append(
            make_training_row(
                prompt=prompt,
                answer=gold_answer,
                evidence=answer_evidence(answer),
                data_source="qasper_evidence",
                index=str(question_ids[idx] if idx < len(question_ids) else idx),
                split=split,
                extra_info={"paper_id": str(paper.get("id", ""))},
            )
        )
    return rows


def load_qasper(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to download QASPER: pip install datasets") from exc
    return load_dataset(dataset_name, trust_remote_code=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QASPER for VeriSeek evidence reward training.")
    add_common_args(parser)
    parser.add_argument("--dataset_name", default="allenai/qasper")
    parser.add_argument("--max_context_chars", type=int, default=12000)
    args = parser.parse_args()

    dataset = load_qasper(args.dataset_name)
    split_plan = {
        "train": ("train", args.max_train),
        "dev": ("validation", args.max_dev),
        "test": ("test", args.max_test),
    }
    output_dir = Path(args.output_dir)
    for out_split, (source_split, max_rows) in split_plan.items():
        if source_split not in dataset:
            continue
        rows_iter = (
            row
            for paper in dataset[source_split]
            for row in convert_paper(paper, out_split, max_context_chars=args.max_context_chars)
        )
        rows = limit_rows(rows_iter, max_rows)
        write_rows(rows, output_dir / output_name(out_split, args.write_jsonl), args.write_jsonl)
        print(f"wrote {len(rows)} {out_split} rows")


if __name__ == "__main__":
    main()
