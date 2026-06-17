from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
LOCAL_MODEL = "models/Qwen3-4B-Thinking-2507"
PUBLIC_FILES = [
    "README.md",
    "README_CN.md",
    "docs/benchmark_report.md",
    "docs/smoke_training.md",
    "docs/reproduction.md",
    "scripts/train_veriseek_grpo.sh",
    "scripts/train_veriseek_rl_only.sh",
    "scripts/train_veriseek_sft.sh",
    "scripts/train_veriseek_sft_rl.sh",
    "scripts/local_prepare_assets.sh",
    "scripts/remote_smoke_train.sh",
]


class ModelDefaultTests(unittest.TestCase):
    def test_public_files_use_qwen3_as_default_model(self):
        for rel_path in PUBLIC_FILES:
            path = ROOT / rel_path
            self.assertTrue(path.exists(), f"missing {rel_path}")
            text = path.read_text(encoding="utf-8")
            self.assertIn("Qwen3-4B-Thinking-2507", text, rel_path)

    def test_no_public_default_uses_dr_venus_sft(self):
        old_checkpoint = "DR-" + "Venus" + "-4B-" + "SFT"
        forbidden = re.compile(rf"inclusionAI/{old_checkpoint}|{old_checkpoint}")
        for rel_path in PUBLIC_FILES:
            text = (ROOT / rel_path).read_text(encoding="utf-8")
            self.assertIsNone(forbidden.search(text), rel_path)


if __name__ == "__main__":
    unittest.main()
