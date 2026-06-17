import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "eval"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("veriseek_common", ROOT / "data" / "veriseek_common.py")
scifact = load_module("prepare_scifact", ROOT / "data" / "prepare_scifact.py")
qasper = load_module("prepare_qasper", ROOT / "data" / "prepare_qasper.py")
litqa2 = load_module("prepare_litqa2", ROOT / "data" / "prepare_litqa2.py")
metrics = load_module("metrics", ROOT / "eval" / "metrics.py")


class DataPipelineTests(unittest.TestCase):
    def test_make_training_row_stores_ground_truth_as_json_string(self):
        row = common.make_training_row(
            prompt="Claim: A causes B.",
            answer="SUPPORTS",
            evidence=["A causes B."],
            data_source="scifact_evidence",
            index="s1",
            split="train",
        )
        self.assertEqual(row["prompt"], [{"role": "user", "content": "Claim: A causes B."}])
        self.assertEqual(row["data_source"], "scifact_evidence")
        self.assertIsInstance(row["reward_model"]["ground_truth"], str)
        self.assertEqual(json.loads(row["reward_model"]["ground_truth"])["answer"], "SUPPORTS")

    def test_scifact_converter_builds_expected_row(self):
        claim = {
            "id": 1,
            "claim": "Drug A reduces inflammation.",
            "evidence_doc_id": "10",
            "evidence_label": "SUPPORT",
            "evidence_sentences": [0],
        }
        corpus = {"10": {"title": "Drug A", "abstract": ["Drug A reduces inflammation."]}}
        row = scifact.convert_claim(claim, corpus, "train")
        gold = json.loads(row["reward_model"]["ground_truth"])
        self.assertEqual(row["data_source"], "scifact_evidence")
        self.assertEqual(gold["answer"], "SUPPORTS")
        self.assertEqual(gold["evidence"], ["Drug A reduces inflammation."])

    def test_qasper_converter_builds_expected_row(self):
        paper = {
            "id": "p1",
            "title": "Neural Retrieval",
            "abstract": "We study retrieval.",
            "full_text": {"section_name": ["Intro"], "paragraphs": [["Retrieval improves with reranking."]]},
            "qas": {
                "question": ["What improves retrieval?"],
                "question_id": ["q1"],
                "answers": [
                    {
                        "answer": [
                            {
                                "unanswerable": False,
                                "extractive_spans": ["reranking"],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": ["Retrieval improves with reranking."],
                                "highlighted_evidence": ["Retrieval improves with reranking."],
                            }
                        ]
                    }
                ],
            },
        }
        rows = qasper.convert_paper(paper, "train")
        gold = json.loads(rows[0]["reward_model"]["ground_truth"])
        self.assertEqual(rows[0]["data_source"], "qasper_evidence")
        self.assertEqual(gold["answer"], "reranking")
        self.assertEqual(gold["evidence"], ["Retrieval improves with reranking."])

    def test_litqa2_converter_builds_eval_row(self):
        example = {
            "id": "l1",
            "question": "Which choice is correct?",
            "ideal": "A",
            "distractors": ["B", "C"],
            "sources": ["Paper evidence."],
            "subtask": "LitQA2",
        }
        row = litqa2.convert_example(example, "test")
        gold = json.loads(row["reward_model"]["ground_truth"])
        self.assertEqual(row["data_source"], "qasper_evidence")
        self.assertEqual(gold["answer"], "A")

    def test_eval_metrics_from_prediction_records(self):
        record = {
            "prediction": "<answer>SUPPORTS</answer><evidence>[1] Drug A reduces inflammation.</evidence>",
            "ground_truth": json.dumps({"answer": "SUPPORTS", "evidence": ["Drug A reduces inflammation."]}),
            "data_source": "scifact_evidence",
        }
        result = metrics.evaluate_records([record], "scifact")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["label_accuracy"], 1.0)
        self.assertEqual(result["format_success_rate"], 1.0)

    def test_relaxed_scifact_metrics_score_natural_language_predictions(self):
        record = {
            "prediction": "The claim is supported. Drug A reduces inflammation in the trial.",
            "ground_truth": json.dumps(
                {"answer": "SUPPORTS", "evidence": ["Drug A reduces inflammation."]}
            ),
            "data_source": "scifact_evidence",
        }
        result = metrics.evaluate_records([record], "scifact", mode="both")
        self.assertEqual(result["label_accuracy"], 0.0)
        self.assertEqual(result["format_success_rate"], 0.0)
        self.assertEqual(result["relaxed_label_accuracy"], 1.0)
        self.assertGreater(result["relaxed_full_text_evidence_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
