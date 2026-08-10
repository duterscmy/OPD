from __future__ import annotations

import unittest

from opd.behavior_markers import (
    RolloutBehaviorAnalyzer,
    aggregate_occurrence_logs,
    repetition_continuation_candidates,
)


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) + 1 for char in text]

    def decode(self, ids, **kwargs) -> str:
        del kwargs
        chars = []
        for token_id in ids:
            value = int(token_id) - 1
            chars.append(chr(value) if 0 <= value <= 0x10FFFF else "<?>")
        return "".join(chars)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.decode([token_id])


class BehaviorMarkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.analyzer = RolloutBehaviorAnalyzer(self.tokenizer)

    def test_curated_categories_and_positions(self) -> None:
        text = (
            "Let's consider the first route. However, this is incorrect. "
            "Let's try another approach. Therefore, the final answer is "
            r"\boxed{42}."
        )
        eos_id = 1_200_000
        ids = self.tokenizer.encode(text) + [eos_id]
        record = self.analyzer.analyze(
            ids,
            text,
            eos_token_ids=[eos_id],
            repetition_ngram_size=4,
        )

        self.assertTrue(record["categories"]["planning"]["document_hit"])
        self.assertTrue(record["categories"]["self_correction"]["document_hit"])
        self.assertTrue(
            record["categories"]["alternative_approach"]["document_hit"]
        )
        self.assertTrue(record["categories"]["conclusion"]["document_hit"])
        self.assertGreaterEqual(record["categories"]["termination"]["count"], 3)
        self.assertEqual(record["structure"]["boxed"]["count"], 1)
        self.assertIn(
            "<EOS>",
            record["categories"]["termination"]["matched_markers"],
        )
        self.assertEqual(
            record["categories"]["planning"]["first_token_position"], 1
        )

    def test_repetition_continuation_candidates(self) -> None:
        tokens = [1, 2, 3, 4, 9, 1, 2, 3, 4]
        candidates = repetition_continuation_candidates(tokens, 4)
        self.assertEqual(candidates[3], ())
        self.assertEqual(candidates[8], (4,))

        record = self.analyzer.analyze(
            tokens,
            "",
            repetition_ngram_size=4,
        )
        repetition = record["repetition_continuation"]
        self.assertEqual(repetition["actual_continuation_count"], 1)
        self.assertEqual(repetition["first_actual_continuation_position"], 9)

    def test_probability_sets_label_phrase_starts(self) -> None:
        eos_id = 1_200_000
        token_sets = self.analyzer.probability_token_sets(
            eos_token_ids=[eos_id]
        )
        self.assertIn(eos_id, token_sets["marker/eos"])
        self.assertIn(eos_id, token_sets["category/termination"])
        self.assertTrue(token_sets["marker/final_answer"])
        manifest = self.analyzer.manifest(eos_token_ids=[eos_id])
        self.assertIn("first tokens", manifest["measurement"])

    def test_occurrence_aggregation(self) -> None:
        first = self.analyzer.analyze(
            self.tokenizer.encode("Therefore, continue."),
            "Therefore, continue.",
        )
        second = self.analyzer.analyze(
            self.tokenizer.encode("The answer is 3."),
            "The answer is 3.",
        )
        logs = aggregate_occurrence_logs([first, second])
        self.assertEqual(
            logs["behavior_occurrence/conclusion_document_fraction"], 0.5
        )
        self.assertEqual(
            logs["behavior_occurrence/termination_document_fraction"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
