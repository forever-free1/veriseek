import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def make_training_row(
    prompt: str,
    answer: Any,
    evidence: list[str],
    data_source: str,
    index: str,
    split: str,
    ability: str = "scientific evidence QA",
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = {"index": str(index), "split": split}
    if extra_info:
        info.update(extra_info)
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {
            "ground_truth": json.dumps(
                {"answer": str(answer), "evidence": [str(item) for item in evidence if str(item).strip()]},
                ensure_ascii=False,
            )
        },
        "data_source": data_source,
        "ability": ability,
        "extra_info": info,
    }


def limit_rows(rows: Iterable[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(row)
        if max_rows is not None and len(output) >= max_rows:
            break
    return output


def write_rows(rows: list[dict[str, Any]], output_path: Path, write_jsonl: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".jsonl":
        write_jsonl = True
    if write_jsonl:
        with output_path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas and pyarrow are required to write parquet files") from exc

    pd.DataFrame(rows).to_parquet(output_path, index=False)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output_dir", required=True, help="Directory for processed train/dev/test files.")
    parser.add_argument("--max_train", type=int, default=None, help="Optional maximum train rows.")
    parser.add_argument("--max_dev", type=int, default=None, help="Optional maximum dev rows.")
    parser.add_argument("--max_test", type=int, default=None, help="Optional maximum test rows.")
    parser.add_argument("--write_jsonl", action="store_true", help="Write JSONL instead of parquet.")
    return parser


def output_name(split: str, write_jsonl: bool) -> str:
    return f"{split}.jsonl" if write_jsonl else f"{split}.parquet"
