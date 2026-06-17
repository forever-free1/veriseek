import argparse
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from veriseek_common import add_common_args, limit_rows, make_training_row, output_name, write_rows


LABEL_MAP = {
    "SUPPORT": "SUPPORTS",
    "SUPPORTS": "SUPPORTS",
    "REFUTE": "REFUTES",
    "REFUTES": "REFUTES",
    "CONTRADICT": "REFUTES",
    "CONTRADICTS": "REFUTES",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
    "NEI": "NOT_ENOUGH_INFO",
}

SCIFACT_TARBALL_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def normalize_label(label: Any) -> str:
    return LABEL_MAP.get(str(label).upper().replace(" ", "_"), str(label).upper().replace(" ", "_"))


def build_corpus_index(corpus_split) -> dict[str, dict[str, Any]]:
    index = {}
    for doc in corpus_split:
        index[str(doc.get("doc_id"))] = doc
    return index


def _first_official_evidence(claim: dict[str, Any]) -> tuple[str, str, list[int]]:
    evidence = claim.get("evidence") or {}
    for doc_id, evidence_sets in evidence.items():
        if evidence_sets:
            first = evidence_sets[0]
            return str(doc_id), first.get("label", "NOT_ENOUGH_INFO"), first.get("sentences", [])
    cited = claim.get("cited_doc_ids") or []
    doc_id = str(cited[0]) if cited else ""
    return doc_id, "NOT_ENOUGH_INFO", []


def _abstract_sentences(doc: dict[str, Any] | None) -> list[str]:
    if not doc:
        return []
    abstract = doc.get("abstract", [])
    if isinstance(abstract, str):
        return [abstract]
    return [str(sentence) for sentence in abstract]


def convert_claim(claim: dict[str, Any], corpus: dict[str, dict[str, Any]], split: str) -> dict[str, Any]:
    if "evidence_doc_id" in claim or "evidence_label" in claim:
        doc_id = str(claim.get("evidence_doc_id") or (claim.get("cited_doc_ids") or [""])[0])
        label = claim.get("evidence_label", "NOT_ENOUGH_INFO")
        evidence_indices = claim.get("evidence_sentences") or []
    else:
        doc_id, label, evidence_indices = _first_official_evidence(claim)
    doc = corpus.get(doc_id, {})
    abstract = _abstract_sentences(doc)
    evidence = [abstract[i] for i in evidence_indices if isinstance(i, int) and 0 <= i < len(abstract)]
    title = doc.get("title", "")
    context = "\n".join(f"[{i}] {sentence}" for i, sentence in enumerate(abstract))
    prompt = f"""You are a scientific research agent.

Decide whether the claim is SUPPORTS, REFUTES, or NOT_ENOUGH_INFO.
Return your response using the exact format:

<answer>
SUPPORTS / REFUTES / NOT_ENOUGH_INFO
</answer>

<evidence>
[1] evidence sentence
</evidence>

Claim:
{claim.get("claim", "")}

Paper title:
{title}

Available abstract sentences:
{context}
"""
    return make_training_row(
        prompt=prompt,
        answer=normalize_label(label),
        evidence=evidence,
        data_source="scifact_evidence",
        index=str(claim.get("id", "")),
        split=split,
        extra_info={"doc_id": doc_id},
    )


def load_scifact(dataset_name: str):
    if dataset_name == "official_tarball":
        return load_scifact_official_tarball()
    try:
        from datasets import load_dataset
        claims = load_dataset(dataset_name, "claims")
        corpus = load_dataset(dataset_name, "corpus")
        return claims, corpus
    except Exception as exc:
        print(f"Falling back to official SciFact tarball because Hugging Face loading failed: {exc}")
        return load_scifact_official_tarball()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scifact_official_tarball():
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "data.tar.gz"
        urllib.request.urlretrieve(SCIFACT_TARBALL_URL, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp)
        data_dir = Path(tmp) / "data"
        claims = {
            "train": _read_jsonl(data_dir / "claims_train.jsonl"),
            "dev": _read_jsonl(data_dir / "claims_dev.jsonl"),
            "test": _read_jsonl(data_dir / "claims_test.jsonl"),
        }
        corpus = {"train": _read_jsonl(data_dir / "corpus.jsonl")}
        return claims, corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SciFact for VeriSeek evidence reward training.")
    add_common_args(parser)
    parser.add_argument("--dataset_name", default="allenai/scifact")
    args = parser.parse_args()

    claims, corpus = load_scifact(args.dataset_name)
    corpus_index = build_corpus_index(corpus["train"])
    split_plan = {
        "train": ("train", args.max_train),
        "dev": ("validation", args.max_dev),
        "test": ("test", args.max_test),
    }
    output_dir = Path(args.output_dir)
    for out_split, (source_split, max_rows) in split_plan.items():
        if source_split not in claims and source_split == "validation" and "dev" in claims:
            source_split = "dev"
        if source_split not in claims:
            continue
        rows = limit_rows((convert_claim(item, corpus_index, out_split) for item in claims[source_split]), max_rows)
        write_rows(rows, output_dir / output_name(out_split, args.write_jsonl), args.write_jsonl)
        print(f"wrote {len(rows)} {out_split} rows")


if __name__ == "__main__":
    main()
