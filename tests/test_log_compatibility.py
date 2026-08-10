from __future__ import annotations

import unittest
from pathlib import Path


class LegacyLogCompatibilityTests(unittest.TestCase):
    def test_rollout_console_lines_remain_in_original_order(self) -> None:
        source = (Path(__file__).parents[1] / "opd" / "trainer.py").read_text(
            encoding="utf-8"
        )
        legacy_fragments = [
            "【student rollout length】",
            "【student raw rollout length】",
            "【student rollout stop reason】",
            "【student rollout stop token】",
            "【student rollout hit horizon】",
            "【student raw rollout hit horizon】",
            "【student rollout truncated after boxed answer】",
            "【student rollout appended EOS】",
            "【student raw rollout repeated ",
            "【student effective rollout repeated ",
            "【student rollout boxed count】",
            "【student effective rollout boxed count】",
        ]
        positions = [source.index(fragment) for fragment in legacy_fragments]
        self.assertEqual(positions, sorted(positions))
        behavior_position = source.index("[student behavior]")
        self.assertGreater(behavior_position, positions[-1])

    def test_existing_length_jsonl_keys_remain_present(self) -> None:
        source = (
            Path(__file__).parents[1] / "opd" / "adaptive_trainer.py"
        ).read_text(encoding="utf-8")
        keys = [
            '"student_rollout_length"',
            '"student_raw_rollout_length"',
            '"student_hit_horizon"',
            '"student_raw_hit_horizon"',
            '"student_repeated_ngram_ratio"',
            '"student_effective_repeated_ngram_ratio"',
        ]
        for key in keys:
            self.assertIn(key, source)


if __name__ == "__main__":
    unittest.main()
