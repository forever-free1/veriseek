import json
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REWARD_DIR = ROOT / "RL" / "verl" / "utils" / "reward_score"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _deprecated(_replacement=""):
    def decorator(obj):
        return obj

    return decorator


sys.modules.setdefault("verl", types.ModuleType("verl"))
sys.modules.setdefault("verl.utils", types.ModuleType("verl.utils"))
import_utils = types.ModuleType("verl.utils.import_utils")
import_utils.deprecated = _deprecated
sys.modules["verl.utils.import_utils"] = import_utils

evidence_reward = _load_module(
    "verl.utils.reward_score.evidence_reward",
    REWARD_DIR / "evidence_reward.py",
)
reward_score = _load_module("verl.utils.reward_score", REWARD_DIR / "__init__.py")

compute_score = evidence_reward.compute_score
conciseness_reward = evidence_reward.conciseness_reward
extract_answer = evidence_reward.extract_answer
extract_evidence = evidence_reward.extract_evidence
format_reward = evidence_reward.format_reward
normalize_text = evidence_reward.normalize_text
token_f1 = evidence_reward.token_f1
default_compute_score = reward_score.default_compute_score


class EvidenceRewardTests(unittest.TestCase):
    def test_extracts_answer_and_evidence_blocks(self):
        text = """
<answer>
SUPPORTS
</answer>

<evidence>
[1] A randomized trial reduced symptoms.
[2] The effect was significant.
</evidence>
"""
        self.assertEqual(extract_answer(text), "SUPPORTS")
        self.assertEqual(
            extract_evidence(text),
            ["A randomized trial reduced symptoms.", "The effect was significant."],
        )

    def test_normalization_and_token_f1_are_deterministic(self):
        self.assertEqual(normalize_text(" The, Trial! "), "the trial")
        self.assertAlmostEqual(token_f1("the trial reduced symptoms", "trial reduced symptoms"), 6 / 7)

    def test_format_and_conciseness_rewards(self):
        self.assertEqual(format_reward("<answer>x</answer><evidence>[1] y</evidence>"), 1.0)
        self.assertEqual(format_reward("<answer>x</answer>"), 0.5)
        self.assertEqual(conciseness_reward(["a"]), 1.0)
        self.assertEqual(conciseness_reward(["a", "b", "c", "d", "e"]), 1.0)
        self.assertEqual(conciseness_reward([]), 0.0)
        self.assertEqual(conciseness_reward(["a", "b", "c", "d", "e", "f"]), 0.0)

    def test_scifact_reward_uses_label_accuracy_and_evidence_f1(self):
        prediction = """
<answer>
supports
</answer>
<evidence>
[1] Drug A reduced inflammation in mice.
</evidence>
"""
        gold = json.dumps(
            {
                "answer": "SUPPORTS",
                "evidence": ["Drug A reduced inflammation in mice."],
            }
        )
        self.assertEqual(compute_score(prediction, gold, "scifact_evidence"), 1.0)

    def test_scifact_reward_caps_supported_claims_without_evidence(self):
        prediction = """
<answer>
SUPPORTS
</answer>
<evidence>
</evidence>
"""
        gold = json.dumps(
            {
                "answer": "SUPPORTS",
                "evidence": ["Drug A reduced inflammation in mice."],
            }
        )
        self.assertLessEqual(compute_score(prediction, gold, "scifact_evidence"), 0.25)

    def test_scifact_reward_penalizes_wrong_nei_on_supported_claims(self):
        prediction = """
<answer>
NOT_ENOUGH_INFO
</answer>
<evidence>
</evidence>
"""
        gold = json.dumps(
            {
                "answer": "SUPPORTS",
                "evidence": ["Drug A reduced inflammation in mice."],
            }
        )
        self.assertLessEqual(compute_score(prediction, gold, "scifact_evidence"), 0.05)

    def test_scifact_reward_rewards_correct_nei_with_empty_evidence(self):
        prediction = """
<answer>
NOT_ENOUGH_INFO
</answer>
<evidence>
</evidence>
"""
        gold = json.dumps({"answer": "NOT_ENOUGH_INFO", "evidence": []})
        self.assertEqual(compute_score(prediction, gold, "scifact_evidence"), 1.0)

    def test_qasper_reward_uses_answer_f1_and_evidence_f1(self):
        prediction = """
<answer>
neural reranking improves retrieval
</answer>
<evidence>
[1] The paper reports neural reranking improves retrieval quality.
</evidence>
"""
        gold = json.dumps(
            {
                "answer": "neural reranking improves retrieval quality",
                "evidence": ["The paper reports neural reranking improves retrieval quality."],
            }
        )
        score = compute_score(prediction, gold, "qasper_evidence")
        self.assertGreater(score, 0.9)
        self.assertLess(score, 1.0)

    def test_reward_routing_supports_evidence_sources(self):
        prediction = "<answer>SUPPORTS</answer><evidence>[1] A causes B.</evidence>"
        gold = json.dumps({"answer": "SUPPORTS", "evidence": ["A causes B."]})
        self.assertEqual(
            default_compute_score("scifact_evidence", "", prediction, gold),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
