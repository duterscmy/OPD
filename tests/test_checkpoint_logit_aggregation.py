from __future__ import annotations

import unittest

from opd.behavior_markers import RolloutBehaviorAnalyzer, aggregate_occurrence_logs
from opd.checkpoint_logit_stats import (
    SUMMED_TOKEN_METRICS,
    aggregate_category_rows,
    aggregate_marker_signal_rows,
    attach_correctness_transitions,
    make_sample_diagnostic,
)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) + 1 for character in text]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(token_id) - 1) for token_id in token_ids if token_id)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return chr(int(token_id) - 1)


class CheckpointLogitAggregationTest(unittest.TestCase):
    def test_occurrence_density_is_length_normalized(self) -> None:
        records = [
            {
                "token_count": 100,
                "categories": {
                    "planning": {
                        "count": 2,
                        "document_hit": True,
                        "first_relative_position": 0.1,
                    }
                },
                "repetition_continuation": {},
            },
            {
                "token_count": 300,
                "categories": {
                    "planning": {
                        "count": 2,
                        "document_hit": True,
                        "first_relative_position": 0.2,
                    }
                },
                "repetition_continuation": {},
            },
        ]
        logs = aggregate_occurrence_logs(records)
        self.assertAlmostEqual(
            logs["behavior_occurrence/planning_density_per_1k"], 10.0
        )
        self.assertAlmostEqual(
            logs["behavior_occurrence/planning_mean_document_density_per_1k"],
            (20.0 + 2.0 / 300.0 * 1000.0) / 2.0,
        )

    def test_marker_span_labels_cover_phrase_and_keep_other(self) -> None:
        tokenizer = CharacterTokenizer()
        analyzer = RolloutBehaviorAnalyzer(tokenizer)
        text = "Let's check."
        token_ids = tokenizer.encode(text)
        labels, rows = analyzer.exclusive_marker_span_labels(
            token_ids,
            text,
            category_priority=("verification", "planning"),
        )
        self.assertTrue(any(row["category"] == "planning" for row in rows))
        self.assertTrue(any(row["category"] == "verification" for row in rows))
        self.assertEqual(labels[:5], ["planning"] * 5)
        check_start = text.index("check")
        self.assertEqual(labels[check_start : check_start + 5], ["verification"] * 5)
        self.assertIn("other", labels)

    def _sample(self, checkpoint: str, step: int, correct: bool) -> dict:
        length = 4
        metrics = {name: [0.0] * length for name in SUMMED_TOKEN_METRICS}
        metrics.update(
            {
                "configured_loss": [1.0, 2.0, 3.0, 4.0],
                "reverse_loss": [1.0, 2.0, 3.0, 4.0],
                "forward_loss": [4.0, 3.0, 2.0, 1.0],
                "signed_advantage": [-1.0, 2.0, -3.0, 4.0],
                "absolute_advantage": [1.0, 2.0, 3.0, 4.0],
                "logit_gradient_proxy": [1.0, 1.0, 2.0, 2.0],
                "training_weighted_logit_gradient_proxy": [0.25, 0.25, 0.5, 0.5],
            }
        )
        record, _ = make_sample_diagnostic(
            metadata={
                "view": "on_policy",
                "scoring_checkpoint": checkpoint,
                "scoring_checkpoint_step": step,
                "source_checkpoint": checkpoint,
                "sample_id": "sample-1",
                "dataset_index": 1,
                "rollout_length": length,
                "is_correct": correct,
            },
            category_labels=["planning", "planning", "other", "termination"],
            marker_rows=[
                {
                    "category": "planning",
                    "response_position": 1,
                    "response_end_position": 2,
                    "markers": ["let's"],
                },
                {
                    "category": "termination",
                    "response_position": 4,
                    "response_end_position": 4,
                    "markers": ["<EOS>"],
                },
            ],
            token_metrics=metrics,
            probability_summary={"sets": {}},
            loss_normalization="per_sequence",
            sequence_loss_weight=1.0,
            terminal_metrics={"terminal_student_entropy": 1.0},
        )
        return record

    def test_category_mass_shares_sum_to_one(self) -> None:
        record = self._sample("checkpoint-25", 25, False)
        sample_share = sum(
            item["configured_loss_raw_mass_share"]
            for item in record["categories"].values()
        )
        self.assertAlmostEqual(sample_share, 1.0)
        rows = aggregate_category_rows([record])
        all_rows = [
            row
            for row in rows
            if row["subset_type"] == "all" and row["subset"] == "all"
        ]
        self.assertAlmostEqual(
            sum(row["configured_loss_training_mass_share"] for row in all_rows),
            1.0,
        )
        planning = next(row for row in all_rows if row["category"] == "planning")
        self.assertAlmostEqual(planning["mean_signed_advantage"], 0.5)
        self.assertAlmostEqual(planning["positive_advantage_fraction"], 0.5)

    def test_correctness_transition_uses_first_checkpoint(self) -> None:
        records = [
            self._sample("checkpoint-25", 25, False),
            self._sample("checkpoint-50", 50, True),
        ]
        attach_correctness_transitions(records)
        self.assertEqual(records[0]["correctness_transition"], "wrong_to_wrong")
        self.assertEqual(records[1]["correctness_transition"], "wrong_to_correct")

    def test_training_mass_respects_per_sequence_normalization(self) -> None:
        def build(
            sample_id: str,
            labels: list[str],
            losses: list[float],
            sequence_weight: float = 1.0,
        ) -> dict:
            metrics = {
                name: [0.0] * len(labels) for name in SUMMED_TOKEN_METRICS
            }
            metrics["configured_loss"] = losses
            record, _ = make_sample_diagnostic(
                metadata={
                    "view": "on_policy",
                    "scoring_checkpoint": "checkpoint-25",
                    "scoring_checkpoint_step": 25,
                    "source_checkpoint": "checkpoint-25",
                    "sample_id": sample_id,
                    "rollout_length": len(labels),
                    "is_correct": True,
                },
                category_labels=labels,
                marker_rows=[],
                token_metrics=metrics,
                probability_summary={"sets": {}},
                loss_normalization="per_sequence",
                sequence_loss_weight=sequence_weight,
                terminal_metrics={},
            )
            return record

        records = [
            build("short", ["planning", "other"], [10.0, 0.0]),
            build("long", ["other"] * 4, [1.0, 1.0, 1.0, 1.0]),
        ]
        rows = [
            row
            for row in aggregate_category_rows(records)
            if row["subset_type"] == "all" and row["subset"] == "all"
        ]
        planning = next(row for row in rows if row["category"] == "planning")
        self.assertAlmostEqual(planning["configured_loss_raw_mass_share"], 10.0 / 14.0)
        self.assertAlmostEqual(
            planning["configured_loss_training_mass_share"], 5.0 / 6.0
        )

        zero_weight_rows = [
            row
            for row in aggregate_category_rows(
                [
                    build(
                        "truncated",
                        ["planning", "other"],
                        [10.0, 0.0],
                        sequence_weight=0.0,
                    ),
                    build("kept", ["other"] * 4, [1.0, 1.0, 1.0, 1.0]),
                ]
            )
            if row["subset_type"] == "all" and row["subset"] == "all"
        ]
        truncated_planning = next(
            row for row in zero_weight_rows if row["category"] == "planning"
        )
        self.assertAlmostEqual(
            truncated_planning["configured_loss_training_mass_share"], 0.0
        )

    def test_individual_marker_signal_uses_all_tokens_as_density_denominator(self) -> None:
        record = self._sample("checkpoint-25", 25, True)
        marker = {
            "view": "on_policy",
            "scoring_checkpoint": "checkpoint-25",
            "scoring_checkpoint_step": 25,
            "source_checkpoint": "checkpoint-25",
            "sample_id": "sample-1",
            "category": "planning",
            "marker": "let's",
            "relative_position": 0.25,
            "signed_advantage": 2.0,
        }
        rows = [
            row
            for row in aggregate_marker_signal_rows([record], [marker])
            if row["subset_type"] == "all"
        ]
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["event_density_per_1k"], 250.0)
        self.assertAlmostEqual(rows[0]["mean_signed_advantage"], 2.0)


if __name__ == "__main__":
    unittest.main()
